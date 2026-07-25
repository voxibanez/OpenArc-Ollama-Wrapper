package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/huggingface"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/lifecycle"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

func TestOpenAIModelsReturnsManifestModelsWithoutDependencies(t *testing.T) {
	server, hfCalls, oaCalls := testServer(t, `
models:
  - name: one
    hf_repo: OpenVINO/one-ov
  - name: two
    aliases: [two:latest]
    hf_repo: OpenVINO/two-ov
`, http.NotFoundHandler(), http.NotFoundHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || len(out.Data) != 2 {
		t.Fatalf("response = %#v", out)
	}
	if out.Data[0]["id"] != "one" || out.Data[1]["id"] != "two" {
		t.Fatalf("models = %#v", out.Data)
	}
	if hfCalls.Load() != 0 || oaCalls.Load() != 0 {
		t.Fatalf("models should not call dependencies, hf=%d oa=%d", hfCalls.Load(), oaCalls.Load())
	}
}

func TestOpenAIModelResolvesAlias(t *testing.T) {
	server, _, _ := testServer(t, `
models:
  - name: vision
    aliases: [vision:latest, hf.co/example/vision]
    hf_repo: OpenVINO/vision-ov
`, http.NotFoundHandler(), http.NotFoundHandler())

	for _, path := range []string{"/v1/models/vision:latest", "/v1/models/hf.co/example/vision"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		server.Routes().ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("path %s: status = %d, body = %s", path, rec.Code, rec.Body.String())
		}
		var out map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if out["id"] != "vision" || out["object"] != "model" {
			t.Fatalf("path %s: response = %#v", path, out)
		}
	}
}

func TestOpenAIUnknownModelUsesOpenAIErrorShape(t *testing.T) {
	server, hfCalls, oaCalls := testServer(t, `
models:
  - name: known
    hf_repo: OpenVINO/known-ov
`, http.NotFoundHandler(), http.NotFoundHandler())

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`),
	)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	apiErr, ok := out["error"].(map[string]any)
	if !ok || !strings.Contains(fmt.Sprint(apiErr["message"]), "models.yaml") {
		t.Fatalf("response = %#v", out)
	}
	if hfCalls.Load() != 0 || oaCalls.Load() != 0 {
		t.Fatalf("unknown model should not call dependencies, hf=%d oa=%d", hfCalls.Load(), oaCalls.Load())
	}
}

func TestOpenAIChatLoadsOnceAndReusesModel(t *testing.T) {
	var loads atomic.Int32
	var chats atomic.Int32
	var mu sync.Mutex
	var proxied []map[string]any

	server := newLocalOpenAITestServer(t, "vlm", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			loads.Add(1)
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/chat/completions":
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
			}
			mu.Lock()
			proxied = append(proxied, payload)
			mu.Unlock()
			chats.Add(1)
			writeJSON(w, http.StatusOK, map[string]any{
				"id":     "chatcmpl-test",
				"object": "chat.completion",
				"model":  "vision",
				"choices": []any{map[string]any{
					"message": map[string]any{"role": "assistant", "content": "A receipt."},
				}},
			})
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	body := `{
		"model":"vision:latest",
		"messages":[{"role":"user","content":[
			{"type":"text","text":"describe this"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}
		]}]
	}`
	for range 2 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		server.Routes().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
		}
		var out map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if out["id"] != "chatcmpl-test" || out["object"] != "chat.completion" {
			t.Fatalf("response = %#v", out)
		}
	}

	if loads.Load() != 1 || chats.Load() != 2 {
		t.Fatalf("loads=%d chats=%d", loads.Load(), chats.Load())
	}
	mu.Lock()
	defer mu.Unlock()
	if proxied[0]["model"] != "vision" {
		t.Fatalf("proxied model = %#v", proxied[0]["model"])
	}
	parts := proxied[0]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	imageURL := parts[1].(map[string]any)["image_url"].(map[string]any)["url"]
	if imageURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image url = %#v", imageURL)
	}
}

func TestOpenAIKeepAliveHeaderUnloadsAfterRequest(t *testing.T) {
	var unloads atomic.Int32
	var proxied map[string]any
	server := newLocalOpenAITestServer(t, "llm", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/openarc/unload":
			unloads.Add(1)
			writeJSON(w, http.StatusOK, map[string]any{"status": "unloading"})
		case "/v1/chat/completions":
			if err := json.NewDecoder(r.Body).Decode(&proxied); err != nil {
				t.Error(err)
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"id":      "chatcmpl-test",
				"object":  "chat.completion",
				"choices": []any{},
			})
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"vision","messages":[{"role":"user","content":"hi"}]}`),
	)
	req.Header.Set(openAIKeepAliveHeader, "0")
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if unloads.Load() != 1 {
		t.Fatalf("unloads = %d", unloads.Load())
	}
	if _, exists := proxied["keep_alive"]; exists {
		t.Fatalf("keep_alive leaked to OpenArc: %#v", proxied)
	}
}

func TestOpenAIStreamingPassesThroughSSE(t *testing.T) {
	server := newLocalOpenAITestServer(t, "llm", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"vision","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
	)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if contentType := rec.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("content type = %q", contentType)
	}
	if !strings.Contains(rec.Body.String(), `"id":"chatcmpl-stream"`) ||
		!strings.HasSuffix(rec.Body.String(), "data: [DONE]\n\n") {
		t.Fatalf("body = %q", rec.Body.String())
	}
}

func TestOpenAIStreamingUpstreamErrorKeepsHTTPErrorStatus(t *testing.T) {
	server := newLocalOpenAITestServer(t, "llm", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/chat/completions":
			http.Error(w, "generation failed", http.StatusInternalServerError)
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"vision","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
	)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.HasPrefix(rec.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("content type = %q", rec.Header().Get("Content-Type"))
	}
}

func TestOpenAICompletionsPreservesResponse(t *testing.T) {
	server := newLocalOpenAITestServer(t, "llm", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/completions":
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
			}
			if payload["model"] != "vision" || payload["prompt"] != "Hello" {
				t.Errorf("payload = %#v", payload)
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"id":     "cmpl-test",
				"object": "text_completion",
				"model":  "vision",
				"choices": []any{map[string]any{
					"text": " world",
				}},
			})
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"model":"vision:latest","prompt":"Hello"}`),
	)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["id"] != "cmpl-test" || out["object"] != "text_completion" {
		t.Fatalf("response = %#v", out)
	}
}

func TestOpenAIEmbeddingsUsesLifecycleAndPreservesResponse(t *testing.T) {
	server := newLocalOpenAITestServer(t, "emb", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/embeddings":
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
			}
			if payload["model"] != "vision" || payload["input"] != "hello" {
				t.Errorf("payload = %#v", payload)
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"object": "list",
				"model":  "vision",
				"data":   []any{map[string]any{"object": "embedding", "index": 0, "embedding": []any{0.1, 0.2}}},
			})
		default:
			t.Errorf("unexpected OpenArc path %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
		}
	}))

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/embeddings",
		strings.NewReader(`{"model":"vision:latest","input":"hello"}`),
	)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["object"] != "list" || out["model"] != "vision" {
		t.Fatalf("response = %#v", out)
	}
}

func newLocalOpenAITestServer(t *testing.T, modelType string, oaHandler http.Handler) *Server {
	t.Helper()
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "model")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "openvino_model.xml"), []byte("<xml/>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "openvino_model.bin"), []byte("bin"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(dir, "models.yaml")
	manifestYAML := `models:
  - name: vision
    aliases: [vision:latest]
    hf_repo: OpenVINO/vision
    model_path: ` + modelDir + `
    model_type: ` + modelType + `
`
	if err := os.WriteFile(manifestPath, []byte(manifestYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := config.LoadManifest(manifestPath, filepath.Join(dir, "models"), "CPU")
	if err != nil {
		t.Fatal(err)
	}

	oaSrv := httptest.NewServer(oaHandler)
	t.Cleanup(oaSrv.Close)
	hfSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("Hugging Face should not be called for a local model")
		http.Error(w, "unexpected HF call", http.StatusInternalServerError)
	}))
	t.Cleanup(hfSrv.Close)

	oaClient := openarc.NewClient(oaSrv.URL, "", oaSrv.Client())
	manager := lifecycle.NewManager(manifest, oaClient, lifecycle.Options{
		MaxLoadedModels:      1,
		DefaultKeepAlive:     time.Minute,
		CheckInterval:        time.Hour,
		DownloadPollInterval: time.Millisecond,
	})
	hfClient := huggingface.NewClient(hfSrv.URL, "", hfSrv.Client())
	return NewServer(manifest, manager, oaClient, hfClient, 16<<20)
}

package ollama

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/huggingface"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/lifecycle"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

func testServer(t *testing.T, manifestYAML string, hfHandler http.Handler, oaHandler http.Handler) (*Server, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "models.yaml")
	if err := os.WriteFile(manifestPath, []byte(manifestYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := config.LoadManifest(manifestPath, filepath.Join(dir, "models"), "CPU")
	if err != nil {
		t.Fatal(err)
	}

	hfCalls := &atomic.Int32{}
	oaCalls := &atomic.Int32{}
	hfSrv := httptest.NewServer(countingHandler(hfCalls, hfHandler))
	oaSrv := httptest.NewServer(countingHandler(oaCalls, oaHandler))
	t.Cleanup(hfSrv.Close)
	t.Cleanup(oaSrv.Close)

	oaClient := openarc.NewClient(oaSrv.URL, "", oaSrv.Client())
	manager := lifecycle.NewManager(manifest, oaClient, lifecycle.Options{
		MaxLoadedModels:      1,
		DefaultKeepAlive:     time.Minute,
		CheckInterval:        time.Hour,
		DownloadPollInterval: time.Millisecond,
	})
	hfClient := huggingface.NewClient(hfSrv.URL, "", hfSrv.Client())
	return NewServer(manifest, manager, oaClient, hfClient, 16<<20), hfCalls, oaCalls
}

func countingHandler(count *atomic.Int32, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count.Add(1)
		next.ServeHTTP(w, r)
	})
}

func TestTagsReturnsAllManifestModelsWithoutOpenArc(t *testing.T) {
	server, hfCalls, oaCalls := testServer(t, `
models:
  - name: one
    hf_repo: OpenVINO/one-ov
  - name: two
    aliases: [two:latest]
    hf_repo: OpenVINO/two-ov
`, http.NotFoundHandler(), http.NotFoundHandler())

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Models) != 2 {
		t.Fatalf("models len = %d", len(out.Models))
	}
	if hfCalls.Load() != 0 || oaCalls.Load() != 0 {
		t.Fatalf("tags should not call dependencies, hf=%d oa=%d", hfCalls.Load(), oaCalls.Load())
	}
}

func TestUnknownModelReturns404BeforeDependencies(t *testing.T) {
	server, hfCalls, oaCalls := testServer(t, `
models:
  - name: known
    hf_repo: OpenVINO/known-ov
`, http.NotFoundHandler(), http.NotFoundHandler())

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"missing","prompt":"hi"}`))
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if hfCalls.Load() != 0 || oaCalls.Load() != 0 {
		t.Fatalf("unknown model should not call dependencies, hf=%d oa=%d", hfCalls.Load(), oaCalls.Load())
	}
}

func TestRequestBodyLimit(t *testing.T) {
	server, _, _ := testServer(t, `
models:
  - name: known
    hf_repo: OpenVINO/known-ov
`, http.NotFoundHandler(), http.NotFoundHandler())
	server.maxRequestBytes = 32

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"known","prompt":"this request is intentionally too large"}`))
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}

func TestPullRejectsNonOpenVINOBeforeDownload(t *testing.T) {
	hfHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"id": "OpenVINO/bad",
			"siblings": []map[string]any{
				{"rfilename": "model.gguf"},
				{"rfilename": "README.md"},
			},
		})
	})
	oaHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("OpenArc should not be called when HF metadata is incompatible")
	})
	server, _, _ := testServer(t, `
models:
  - name: bad
    hf_repo: OpenVINO/bad
`, hfHandler, oaHandler)

	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"name":"bad","stream":false}`))
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "OpenVINO") {
		t.Fatalf("expected OpenVINO error, got %s", rec.Body.String())
	}
}

func TestToOllamaChatResponseSeparatesThinkingTags(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content": "<think>\nchecking\n</think>\nHello",
				},
			},
		},
	}

	resp := toOllamaResponse(out, "deepseek-r1:7b", "chat")
	msg := resp["message"].(map[string]any)

	if msg["thinking"] != "\nchecking\n" {
		t.Fatalf("thinking = %#v", msg["thinking"])
	}
	if msg["content"] != "\nHello" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

func TestToOllamaChatResponseInfersMissingThinkStart(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content": "I should answer casually.\n</think>\nHey!",
				},
			},
		},
	}

	resp := toOllamaResponse(out, "deepseek-r1:7b", "chat")
	msg := resp["message"].(map[string]any)

	if msg["thinking"] != "I should answer casually.\n" {
		t.Fatalf("thinking = %#v", msg["thinking"])
	}
	if msg["content"] != "\nHey!" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

func TestToOllamaChatResponseLabelsQwenThinkingProcess(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content": "hinking Process:\nAnalyze the Input\n</think>\nHey!",
				},
			},
		},
	}

	resp := toOllamaResponse(out, "qwen3.5:4b", "chat")
	msg := resp["message"].(map[string]any)

	if msg["thinking"] != "hinking Process:\nAnalyze the Input\n" {
		t.Fatalf("thinking = %#v", msg["thinking"])
	}
	if msg["content"] != "\nHey!" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

func TestToOllamaGenerateResponseUsesStructuredThinking(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"text":              "Answer",
				"reasoning_content": "Reasoning",
			},
		},
	}

	resp := toOllamaResponse(out, "deepseek-r1:7b", "generate")

	if resp["thinking"] != "Reasoning" {
		t.Fatalf("thinking = %#v", resp["thinking"])
	}
	if resp["response"] != "Answer" {
		t.Fatalf("response = %#v", resp["response"])
	}
}

func TestStreamChatResponseSeparatesThinkingAcrossChunks(t *testing.T) {
	normalizer := newReasoningNormalizer("deepseek-r1:7b")
	first := streamChunkToOllama(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "I should answer"}}},
	}, "deepseek-r1:7b", "chat", normalizer)
	if first != nil {
		t.Fatalf("first chunk should be buffered, got %#v", first)
	}

	second := streamChunkToOllama(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": ".\n</think>\nHey"}}},
	}, "deepseek-r1:7b", "chat", normalizer)
	msg := second["message"].(map[string]any)

	if msg["thinking"] != "I should answer.\n" {
		t.Fatalf("thinking = %#v", msg["thinking"])
	}
	if msg["content"] != "\nHey" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

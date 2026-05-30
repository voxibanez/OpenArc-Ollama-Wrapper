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

func TestTagModelAdvertisesVisionCapability(t *testing.T) {
	model := config.Model{
		Name:      "qwen3.5:4b",
		ModelType: "vlm",
		Details: map[string]any{
			"family":   "qwen3_5",
			"families": []any{"qwen3_5"},
		},
	}

	item := tagModel(model)
	details := item["details"].(map[string]any)
	families := details["families"].([]string)
	if len(families) != 2 || families[0] != "qwen3_5" || families[1] != "vision" {
		t.Fatalf("families = %#v", families)
	}
	capabilities := details["capabilities"].(map[string]any)
	if capabilities["vision"] != true {
		t.Fatalf("details capabilities = %#v", capabilities)
	}
	infoCaps := item["info"].(map[string]any)["meta"].(map[string]any)["capabilities"].(map[string]any)
	if infoCaps["vision"] != true {
		t.Fatalf("info capabilities = %#v", infoCaps)
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

func TestToOllamaChatResponseRemovesReplacementCharacter(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content": "Hey there! \uFFFD How can I help?",
				},
			},
		},
	}

	resp := toOllamaResponse(out, "qwen3.5:4b", "chat")
	msg := resp["message"].(map[string]any)

	if msg["content"] != "Hey there!  How can I help?" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

func TestStreamChatResponseRemovesReplacementCharacter(t *testing.T) {
	normalizer := newReasoningNormalizer("llama3.2:3b")
	chunk := streamChunkToOllama(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]any{"content": "Hi \uFFFD there"}}},
	}, "llama3.2:3b", "chat", normalizer)
	msg := chunk["message"].(map[string]any)

	if msg["content"] != "Hi  there" {
		t.Fatalf("content = %#v", msg["content"])
	}
}

func TestTranslateChatImagesToOpenAIContentParts(t *testing.T) {
	req := map[string]any{
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "what is this?",
				"images":  []any{"aGVsbG8="},
			},
		},
	}
	model := &config.Model{Name: "vision", ModelType: "vlm"}

	path, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" {
		t.Fatalf("path = %s", path)
	}
	message := req["messages"].([]any)[0].(map[string]any)
	if _, exists := message["images"]; exists {
		t.Fatal("images should be removed after translation")
	}
	parts := message["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts len = %d", len(parts))
	}
	if parts[0].(map[string]any)["type"] != "text" {
		t.Fatalf("first part = %#v", parts[0])
	}
	imagePart := parts[1].(map[string]any)
	if imagePart["type"] != "image_url" {
		t.Fatalf("image part = %#v", imagePart)
	}
	imageURL := imagePart["image_url"].(map[string]any)["url"]
	if imageURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image url = %#v", imageURL)
	}
}

func TestTranslateChatOpenAIImagePartsAreNormalized(t *testing.T) {
	req := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "what is this?"},
					map[string]any{"type": "image_url", "image_url": "aGVsbG8="},
				},
			},
		},
	}
	model := &config.Model{Name: "vision", ModelType: "vlm"}

	_, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err != nil {
		t.Fatal(err)
	}
	parts := req["messages"].([]any)[0].(map[string]any)["content"].([]any)
	imagePart := parts[1].(map[string]any)
	imageURL := imagePart["image_url"].(map[string]any)["url"]
	if imageURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image url = %#v", imageURL)
	}
}

func TestTranslateChatOpenAIImagePartsRejectNonVisionModel(t *testing.T) {
	req := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "what is this?"},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": "data:image/png;base64,aGVsbG8="},
					},
				},
			},
		},
	}
	model := &config.Model{Name: "text", ModelType: "llm"}

	_, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "not configured as a vision model") {
		t.Fatalf("error = %v", err)
	}
}

func TestTranslateChatImageURLRejectsUnconvertedURL(t *testing.T) {
	req := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "what is this?"},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": "http://openwebui/api/v1/files/1/content"},
					},
				},
			},
		},
	}
	model := &config.Model{Name: "vision", ModelType: "vlm"}

	_, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "data:image") {
		t.Fatalf("error = %v", err)
	}
}

func TestChatEndpointForwardsOpenWebUIOllamaImagesToOpenArcVLM(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "vision-model")
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
    hf_repo: OpenVINO/vision
    model_path: ` + modelDir + `
    model_type: vlm
`
	if err := os.WriteFile(manifestPath, []byte(manifestYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := config.LoadManifest(manifestPath, filepath.Join(dir, "models"), "CPU")
	if err != nil {
		t.Fatal(err)
	}

	var proxied map[string]any
	hfCalls := &atomic.Int32{}
	oaHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openarc/status":
			writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
		case "/openarc/load":
			writeJSON(w, http.StatusOK, map[string]any{"status": "loaded"})
		case "/v1/chat/completions":
			if err := json.NewDecoder(r.Body).Decode(&proxied); err != nil {
				t.Fatal(err)
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"choices": []any{map[string]any{
					"message": map[string]any{"content": "A small image."},
				}},
			})
		default:
			t.Fatalf("unexpected OpenArc path %s", r.URL.Path)
		}
	})
	oaSrv := httptest.NewServer(oaHandler)
	t.Cleanup(oaSrv.Close)
	hfSrv := httptest.NewServer(countingHandler(hfCalls, http.NotFoundHandler()))
	t.Cleanup(hfSrv.Close)
	oaClient := openarc.NewClient(oaSrv.URL, "", oaSrv.Client())
	manager := lifecycle.NewManager(manifest, oaClient, lifecycle.Options{
		MaxLoadedModels:      1,
		DefaultKeepAlive:     time.Minute,
		CheckInterval:        time.Hour,
		DownloadPollInterval: time.Millisecond,
	})
	server := NewServer(manifest, manager, oaClient, huggingface.NewClient(hfSrv.URL, "", hfSrv.Client()), 16<<20)

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{
		"model":"vision",
		"stream":false,
		"messages":[{"role":"user","content":"describe this","images":["aGVsbG8="]}]
	}`))
	rec := httptest.NewRecorder()
	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if hfCalls.Load() != 0 {
		t.Fatalf("local model path should avoid HF metadata call, hf=%d", hfCalls.Load())
	}
	messages := proxied["messages"].([]any)
	user := messages[0].(map[string]any)
	parts := user["content"].([]any)
	if parts[0].(map[string]any)["text"] != "describe this" {
		t.Fatalf("text part = %#v", parts[0])
	}
	imageURL := parts[1].(map[string]any)["image_url"].(map[string]any)["url"]
	if imageURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image url = %#v", imageURL)
	}
}

func TestTranslateImagesRejectsNonVisionModel(t *testing.T) {
	req := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "what is this?", "images": []any{"aGVsbG8="}},
		},
	}
	model := &config.Model{Name: "text", ModelType: "llm"}

	_, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "not configured as a vision model") {
		t.Fatalf("error = %v", err)
	}
}

func TestTranslateGenerateImagesRoutesToChatCompletions(t *testing.T) {
	req := map[string]any{
		"system": "be concise",
		"prompt": "describe",
		"images": []any{"data:image/jpeg;base64,abc"},
	}
	model := &config.Model{Name: "vision", ModelType: "vlm"}

	path, err := translateImages(req, model, "/v1/completions", "generate")
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" {
		t.Fatalf("path = %s", path)
	}
	if _, exists := req["images"]; exists {
		t.Fatal("images should be removed from generate request")
	}
	messages := req["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("messages len = %d", len(messages))
	}
	user := messages[1].(map[string]any)
	parts := user["content"].([]any)
	imagePart := parts[1].(map[string]any)
	imageURL := imagePart["image_url"].(map[string]any)["url"]
	if imageURL != "data:image/jpeg;base64,abc" {
		t.Fatalf("image url = %#v", imageURL)
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

func TestToOllamaGenerateResponseAcceptsChatCompletionShape(t *testing.T) {
	out := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content":           "The image shows a receipt.",
					"reasoning_content": "Inspecting the document.",
				},
			},
		},
	}

	resp := toOllamaResponse(out, "qwen3.5:4b", "generate")

	if resp["thinking"] != "Inspecting the document." {
		t.Fatalf("thinking = %#v", resp["thinking"])
	}
	if resp["response"] != "The image shows a receipt." {
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

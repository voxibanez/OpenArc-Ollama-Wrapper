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

package lifecycle

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

type fakeOpenArc struct {
	loadCalls   int
	unloadCalls int
	status      openarc.Status
}

func (f *fakeOpenArc) Load(context.Context, openarc.ModelLoadConfig) error {
	f.loadCalls++
	return nil
}

func (f *fakeOpenArc) Unload(context.Context, string) error {
	f.unloadCalls++
	return nil
}

func (f *fakeOpenArc) Status(context.Context) (openarc.Status, error) {
	return f.status, nil
}

func (f *fakeOpenArc) StartDownload(context.Context, string) error {
	return nil
}

func (f *fakeOpenArc) Downloads(context.Context) ([]openarc.DownloadTask, error) {
	return nil, nil
}

func TestEnsureLoadedReusesModelAlreadyLoadedInOpenArc(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model")
	if err := os.Mkdir(modelPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelPath, "openvino_model.xml"), []byte("<xml/>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelPath, "openvino_model.bin"), []byte("bin"), 0o644); err != nil {
		t.Fatal(err)
	}

	model := &config.Model{
		Name:      "same-model",
		HFRepo:    "OpenVINO/same-model",
		ModelPath: modelPath,
		Engine:    "ovgenai",
		ModelType: "llm",
		Device:    "GPU.0",
	}
	openArc := &fakeOpenArc{
		status: openarc.Status{
			Models: []openarc.LoadedModel{
				{ModelName: "same-model", Status: "loaded"},
			},
		},
	}
	manager := NewManager(&config.Manifest{Models: []config.Model{*model}}, openArc, Options{
		MaxLoadedModels:  1,
		DefaultKeepAlive: time.Minute,
		CheckInterval:    time.Hour,
	})

	lease, err := manager.EnsureLoaded(context.Background(), model, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lease == nil {
		t.Fatal("expected lease")
	}
	if openArc.loadCalls != 0 {
		t.Fatalf("Load calls = %d, want 0", openArc.loadCalls)
	}
}

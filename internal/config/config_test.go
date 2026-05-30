package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParseDuration(t *testing.T) {
	tests := map[string]time.Duration{
		"0":    0,
		"60":   time.Minute,
		"1m":   time.Minute,
		"1.5s": 1500 * time.Millisecond,
	}
	for input, want := range tests {
		got, err := ParseDuration(input)
		if err != nil {
			t.Fatalf("ParseDuration(%q): %v", input, err)
		}
		if got != want {
			t.Fatalf("ParseDuration(%q) = %v, want %v", input, got, want)
		}
	}
}

func TestLoadManifestResolvesAliasesAndDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "models.yaml")
	err := os.WriteFile(path, []byte(`
models:
  - name: qwen
    aliases: [qwen:latest]
    hf_repo: OpenVINO/qwen-ov
`), 0o644)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := LoadManifest(path, "/models", "GPU.0")
	if err != nil {
		t.Fatal(err)
	}
	model, ok := manifest.Resolve("qwen:latest")
	if !ok {
		t.Fatal("expected alias to resolve")
	}
	if model.ModelPath != "/models/OpenVINO__qwen-ov" {
		t.Fatalf("model path = %q", model.ModelPath)
	}
	if model.Engine != "ovgenai" || model.ModelType != "llm" || model.Device != "GPU.0" {
		t.Fatalf("defaults not applied: %#v", model)
	}
}

package compat

import "testing"

func TestCheckFileListRequiresOpenVINOIR(t *testing.T) {
	if err := CheckFileList([]string{"foo_model.xml", "foo_model.bin", "config.json"}); err != nil {
		t.Fatalf("expected valid OpenVINO files: %v", err)
	}
}

func TestCheckFileListRejectsGGUFOnly(t *testing.T) {
	if err := CheckFileList([]string{"model.gguf", "README.md"}); err == nil {
		t.Fatal("expected GGUF-only repo to be rejected")
	}
}

func TestCheckFileListRejectsMissingBin(t *testing.T) {
	if err := CheckFileList([]string{"foo_model.xml", "config.json"}); err == nil {
		t.Fatal("expected missing *_model.bin to be rejected")
	}
}

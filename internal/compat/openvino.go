package compat

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var ErrNotOpenVINO = errors.New("model does not contain OpenArc-compatible OpenVINO IR files")

func CheckFileList(files []string) error {
	hasXML := false
	hasBIN := false
	onlyRejected := true
	for _, file := range files {
		name := strings.ToLower(filepath.Base(file))
		if strings.Contains(name, "_model.xml") {
			hasXML = true
			onlyRejected = false
		}
		if strings.Contains(name, "_model.bin") {
			hasBIN = true
			onlyRejected = false
		}
		if strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".txt") || strings.HasSuffix(name, ".md") {
			continue
		}
		if !(strings.HasSuffix(name, ".gguf") || strings.HasSuffix(name, ".onnx") || strings.HasSuffix(name, ".safetensors") || strings.HasSuffix(name, ".pt") || strings.HasSuffix(name, ".bin")) {
			onlyRejected = false
		}
	}
	if hasXML && hasBIN {
		return nil
	}
	if onlyRejected {
		return fmt.Errorf("%w: repo appears to be GGUF/ONNX/PyTorch-only", ErrNotOpenVINO)
	}
	return fmt.Errorf("%w: require at least one *_model.xml and one *_model.bin", ErrNotOpenVINO)
}

func CheckLocalPath(path string) error {
	var files []string
	err := filepath.WalkDir(path, func(p string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return CheckFileList(files)
}

func LocalPathLooksPresent(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Manifest struct {
	Models []Model `yaml:"models"`
	byName map[string]*Model
}

type Model struct {
	Name          string         `yaml:"name" json:"name"`
	Aliases       []string       `yaml:"aliases" json:"aliases,omitempty"`
	HFRepo        string         `yaml:"hf_repo" json:"hf_repo"`
	HFRevision    string         `yaml:"hf_revision" json:"hf_revision,omitempty"`
	ModelPath     string         `yaml:"model_path" json:"model_path"`
	Engine        string         `yaml:"engine" json:"engine"`
	ModelType     string         `yaml:"model_type" json:"model_type"`
	Device        string         `yaml:"device" json:"device"`
	RuntimeConfig map[string]any `yaml:"runtime_config" json:"runtime_config,omitempty"`
	Size          int64          `yaml:"size" json:"size,omitempty"`
	Digest        string         `yaml:"digest" json:"digest,omitempty"`
	ModifiedAt    string         `yaml:"modified_at" json:"modified_at,omitempty"`
	Details       map[string]any `yaml:"details" json:"details,omitempty"`
}

func LoadManifest(path, modelRoot, defaultDevice string) (*Manifest, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var manifest Manifest
	if err := yaml.Unmarshal(bytes, &manifest); err != nil {
		return nil, err
	}
	if len(manifest.Models) == 0 {
		return nil, fmt.Errorf("%s contains no models", path)
	}
	manifest.byName = make(map[string]*Model)
	for i := range manifest.Models {
		model := &manifest.Models[i]
		if strings.TrimSpace(model.Name) == "" {
			return nil, fmt.Errorf("models[%d].name is required", i)
		}
		if strings.TrimSpace(model.HFRepo) == "" {
			return nil, fmt.Errorf("model %q requires hf_repo", model.Name)
		}
		if model.Engine == "" {
			model.Engine = "ovgenai"
		}
		if model.ModelType == "" {
			model.ModelType = "llm"
		}
		if model.Device == "" {
			model.Device = defaultDevice
		}
		if model.ModelPath == "" {
			model.ModelPath = filepath.Join(modelRoot, strings.ReplaceAll(model.HFRepo, "/", "__"))
		}
		if model.RuntimeConfig == nil {
			model.RuntimeConfig = map[string]any{}
		}
		if model.ModifiedAt == "" {
			model.ModifiedAt = time.Unix(0, 0).UTC().Format(time.RFC3339)
		}
		if err := manifest.add(model.Name, model); err != nil {
			return nil, err
		}
		for _, alias := range model.Aliases {
			if err := manifest.add(alias, model); err != nil {
				return nil, err
			}
		}
	}
	return &manifest, nil
}

func (m *Manifest) add(name string, model *Model) error {
	key := normalizeName(name)
	if key == "" {
		return nil
	}
	if existing, ok := m.byName[key]; ok && existing.Name != model.Name {
		return fmt.Errorf("model name/alias %q is used by both %q and %q", name, existing.Name, model.Name)
	}
	m.byName[key] = model
	return nil
}

func (m *Manifest) Resolve(name string) (*Model, bool) {
	model, ok := m.byName[normalizeName(name)]
	return model, ok
}

func normalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

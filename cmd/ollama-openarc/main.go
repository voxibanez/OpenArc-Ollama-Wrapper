package main

import (
	"log"
	"net/http"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/huggingface"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/lifecycle"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/ollama"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

func main() {
	cfg, err := config.LoadAppConfig()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	manifest, err := config.LoadManifest(cfg.ManifestPath, cfg.ModelPath, cfg.DefaultDevice)
	if err != nil {
		log.Fatalf("manifest: %v", err)
	}

	httpClient := &http.Client{Timeout: 0}
	openArc := openarc.NewClient(cfg.OpenArcBaseURL, cfg.OpenArcAPIKey, httpClient).
		WithLimits(cfg.MaxResponseBytes, cfg.MaxStreamLineBytes)
	hf := huggingface.NewClient(cfg.HuggingFaceBaseURL, cfg.HuggingFaceToken, &http.Client{Timeout: 30 * time.Second}).
		WithMaxMetadataBytes(cfg.MaxResponseBytes)
	manager := lifecycle.NewManager(manifest, openArc, lifecycle.Options{
		MaxLoadedModels:      cfg.MaxLoadedModels,
		DefaultKeepAlive:     cfg.DefaultKeepAlive,
		CheckInterval:        cfg.IdleCheckInterval,
		DownloadPollInterval: cfg.DownloadPollInterval,
	})

	router := ollama.NewServer(manifest, manager, openArc, hf, cfg.MaxRequestBytes).Routes()

	log.Printf("ollama-openarc listening on %s with %d manifest model(s)", cfg.ListenAddr, len(manifest.Models))
	if err := http.ListenAndServe(cfg.ListenAddr, router); err != nil {
		log.Fatal(err)
	}
}

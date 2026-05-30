package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type AppConfig struct {
	ListenAddr           string
	OpenArcBaseURL       string
	OpenArcAPIKey        string
	HuggingFaceBaseURL   string
	HuggingFaceToken     string
	ManifestPath         string
	ModelPath            string
	DefaultDevice        string
	MaxLoadedModels      int
	DefaultKeepAlive     time.Duration
	IdleCheckInterval    time.Duration
	DownloadPollInterval time.Duration
}

func LoadAppConfig() (AppConfig, error) {
	cfg := AppConfig{
		ListenAddr:           env("LISTEN_ADDR", ":11434"),
		OpenArcBaseURL:       strings.TrimRight(env("OPENARC_BASE_URL", "http://localhost:8000"), "/"),
		OpenArcAPIKey:        os.Getenv("OPENARC_API_KEY"),
		HuggingFaceBaseURL:   strings.TrimRight(env("HUGGINGFACE_BASE_URL", "https://huggingface.co"), "/"),
		HuggingFaceToken:     os.Getenv("HUGGINGFACE_TOKEN"),
		ManifestPath:         env("MODEL_MANIFEST", "models.yaml"),
		ModelPath:            env("MODEL_PATH", "./models"),
		DefaultDevice:        env("OPENARC_DEFAULT_DEVICE", "GPU.0"),
		MaxLoadedModels:      envInt("MAX_LOADED_MODELS", 1),
		DefaultKeepAlive:     envDuration("DEFAULT_KEEP_ALIVE", time.Minute),
		IdleCheckInterval:    envDuration("IDLE_CHECK_INTERVAL", 10*time.Second),
		DownloadPollInterval: envDuration("DOWNLOAD_POLL_INTERVAL", time.Second),
	}
	if cfg.MaxLoadedModels < 1 {
		return cfg, errors.New("MAX_LOADED_MODELS must be at least 1")
	}
	return cfg, nil
}

func env(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		d, err := ParseDuration(v)
		if err == nil {
			return d
		}
	}
	return fallback
}

func ParseDuration(value string) (time.Duration, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, errors.New("empty duration")
	}
	if value == "0" {
		return 0, nil
	}
	if d, err := time.ParseDuration(value); err == nil {
		return d, nil
	}
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), nil
	}
	return 0, fmt.Errorf("invalid duration %q", value)
}

package openarc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Client struct {
	baseURL            string
	apiKey             string
	http               *http.Client
	maxResponseBytes   int64
	maxStreamLineBytes int
}

type ModelLoadConfig struct {
	ModelPath     string         `json:"model_path"`
	ModelName     string         `json:"model_name"`
	ModelType     string         `json:"model_type"`
	Engine        string         `json:"engine"`
	Device        string         `json:"device"`
	RuntimeConfig map[string]any `json:"runtime_config"`
}

type Status struct {
	TotalLoadedModels int           `json:"total_loaded_models"`
	Models            []LoadedModel `json:"models"`
	OpenAIModelNames  []string      `json:"openai_model_names"`
}

type LoadedModel struct {
	ModelName  string `json:"model_name"`
	ModelType  string `json:"model_type"`
	Engine     string `json:"engine"`
	Device     string `json:"device"`
	Status     string `json:"status"`
	TimeLoaded string `json:"time_loaded"`
}

type DownloadTask struct {
	ModelName      string `json:"model_name"`
	TotalSize      int64  `json:"total_size"`
	DownloadedSize int64  `json:"downloaded_size"`
	Status         string `json:"status"`
	Progress       int    `json:"progress"`
	DownloadSpeed  int64  `json:"download_speed"`
	Path           string `json:"path"`
}

func NewClient(baseURL, apiKey string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 0}
	}
	return &Client{
		baseURL:            strings.TrimRight(baseURL, "/"),
		apiKey:             apiKey,
		http:               httpClient,
		maxResponseBytes:   64 << 20,
		maxStreamLineBytes: 2 << 20,
	}
}

func (c *Client) WithLimits(maxResponseBytes int64, maxStreamLineBytes int) *Client {
	if maxResponseBytes > 0 {
		c.maxResponseBytes = maxResponseBytes
	}
	if maxStreamLineBytes > 0 {
		c.maxStreamLineBytes = maxStreamLineBytes
	}
	return c
}

func (c *Client) Load(ctx context.Context, cfg ModelLoadConfig) error {
	return c.doJSON(ctx, http.MethodPost, "/openarc/load", cfg, nil)
}

func (c *Client) Unload(ctx context.Context, modelName string) error {
	return c.doJSON(ctx, http.MethodPost, "/openarc/unload", map[string]string{"model_name": modelName}, nil)
}

func (c *Client) Status(ctx context.Context) (Status, error) {
	var status Status
	err := c.doJSON(ctx, http.MethodGet, "/openarc/status", nil, &status)
	return status, err
}

func (c *Client) StartDownload(ctx context.Context, hfRepo string) error {
	return c.doJSON(ctx, http.MethodPost, "/openarc/downloader", map[string]string{"model_name": hfRepo}, nil)
}

func (c *Client) Downloads(ctx context.Context) ([]DownloadTask, error) {
	var out struct {
		Models []DownloadTask `json:"models"`
	}
	err := c.doJSON(ctx, http.MethodGet, "/openarc/downloader", nil, &out)
	return out.Models, err
}

func (c *Client) DeleteModel(ctx context.Context, modelPath string) error {
	return c.doJSON(ctx, http.MethodDelete, "/openarc/models", map[string]string{"model_path": modelPath}, nil)
}

func (c *Client) ProxyJSON(ctx context.Context, path string, payload any) (map[string]any, error) {
	var out map[string]any
	err := c.doJSON(ctx, http.MethodPost, path, payload, &out)
	return out, err
}

func (c *Client) StreamSSE(ctx context.Context, path string, payload any, onData func(map[string]any) error) error {
	return c.StreamSSEWithStart(ctx, path, payload, nil, onData)
}

func (c *Client) StreamSSEWithStart(
	ctx context.Context,
	path string,
	payload any,
	onStart func(),
	onData func(map[string]any) error,
) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	c.decorate(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("openarc stream failed: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	if onStart != nil {
		onStart()
	}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), c.maxStreamLineBytes)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if err := onData(chunk); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (c *Client) doJSON(ctx context.Context, method, path string, payload any, out any) error {
	var body io.Reader
	if payload != nil {
		bytes, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = strings.NewReader(string(bytes))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return err
	}
	c.decorate(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("openarc %s %s failed: %s: %s", method, path, resp.Status, strings.TrimSpace(string(b)))
	}
	if out == nil {
		io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(io.LimitReader(resp.Body, c.maxResponseBytes)).Decode(out)
}

func (c *Client) decorate(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

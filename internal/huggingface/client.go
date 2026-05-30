package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/compat"
)

type Client struct {
	baseURL          string
	token            string
	http             *http.Client
	maxMetadataBytes int64
}

type ModelInfo struct {
	ID       string    `json:"id"`
	Siblings []Sibling `json:"siblings"`
}

type Sibling struct {
	RFilename string `json:"rfilename"`
	Size      *int64 `json:"size,omitempty"`
}

func NewClient(baseURL, token string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), token: token, http: httpClient, maxMetadataBytes: 16 << 20}
}

func (c *Client) WithMaxMetadataBytes(maxBytes int64) *Client {
	if maxBytes > 0 {
		c.maxMetadataBytes = maxBytes
	}
	return c
}

func (c *Client) ModelInfo(ctx context.Context, repo, revision string) (ModelInfo, error) {
	path := c.baseURL + "/api/models/" + strings.TrimLeft(repo, "/")
	if revision != "" {
		q := url.Values{}
		q.Set("revision", revision)
		path += "?" + q.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, path, nil)
	if err != nil {
		return ModelInfo{}, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return ModelInfo{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return ModelInfo{}, fmt.Errorf("huggingface repo %q not found", repo)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ModelInfo{}, fmt.Errorf("huggingface metadata failed: %s", resp.Status)
	}
	var info ModelInfo
	if err := json.NewDecoder(io.LimitReader(resp.Body, c.maxMetadataBytes)).Decode(&info); err != nil {
		return ModelInfo{}, err
	}
	return info, nil
}

func (c *Client) CheckOpenVINO(ctx context.Context, repo, revision string) error {
	info, err := c.ModelInfo(ctx, repo, revision)
	if err != nil {
		return err
	}
	files := make([]string, 0, len(info.Siblings))
	for _, sibling := range info.Siblings {
		files = append(files, sibling.RFilename)
	}
	return compat.CheckFileList(files)
}

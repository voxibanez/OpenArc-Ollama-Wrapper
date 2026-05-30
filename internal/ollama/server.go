package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/compat"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/huggingface"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/lifecycle"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

type Server struct {
	manifest *config.Manifest
	manager  *lifecycle.Manager
	openarc  *openarc.Client
	hf       *huggingface.Client
}

func NewServer(manifest *config.Manifest, manager *lifecycle.Manager, openArc *openarc.Client, hf *huggingface.Client) *Server {
	return &Server{manifest: manifest, manager: manager, openarc: openArc, hf: hf}
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Get("/", s.version)
	r.Get("/api/version", s.version)
	r.Get("/api/tags", s.tags)
	r.Get("/api/ps", s.ps)
	r.Post("/api/pull", s.pull)
	r.Post("/api/show", s.show)
	r.Delete("/api/delete", s.delete)
	r.Post("/api/chat", s.chat)
	r.Post("/api/generate", s.generate)
	r.Post("/api/embed", s.embed)
	r.Post("/api/embeddings", s.embeddings)
	r.Post("/api/copy", s.copyModel)
	r.Post("/api/create", unsupported)
	r.Post("/api/push", unsupported)
	r.Post("/api/blobs/{digest}", unsupported)
	return r
}

func (s *Server) version(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"version": "0.1.0-openarc"})
}

func (s *Server) tags(w http.ResponseWriter, r *http.Request) {
	models := make([]map[string]any, 0, len(s.manifest.Models))
	for _, model := range s.manifest.Models {
		models = append(models, tagModel(model))
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (s *Server) ps(w http.ResponseWriter, r *http.Request) {
	status, err := s.openarc.Status(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	models := make([]map[string]any, 0, len(status.Models))
	for _, loaded := range status.Models {
		if model, ok := s.manifest.Resolve(loaded.ModelName); ok {
			item := tagModel(*model)
			item["processor"] = loaded.Device
			item["expires_at"] = ""
			item["size_vram"] = item["size"]
			models = append(models, item)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (s *Server) pull(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name   string `json:"name"`
		Model  string `json:"model"`
		Stream *bool  `json:"stream"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	name := firstNonEmpty(req.Name, req.Model)
	model, ok := s.resolve(w, name)
	if !ok {
		return
	}
	if err := s.hf.CheckOpenVINO(r.Context(), model.HFRepo, model.HFRevision); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("pre-download OpenVINO check failed for %s: %w", model.HFRepo, err))
		return
	}
	if req.Stream != nil && !*req.Stream {
		if err := s.openarc.StartDownload(r.Context(), model.HFRepo); err != nil {
			writeError(w, http.StatusBadGateway, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "pull started"})
		return
	}
	s.streamPull(w, r, model)
}

func (s *Server) streamPull(w http.ResponseWriter, r *http.Request, model *config.Model) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, errors.New("streaming not supported"))
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	writeLine(w, map[string]any{"status": "pulling manifest"})
	flusher.Flush()

	if err := s.openarc.StartDownload(r.Context(), model.HFRepo); err != nil {
		writeLine(w, map[string]any{"status": "error", "error": err.Error()})
		flusher.Flush()
		return
	}

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			tasks, err := s.openarc.Downloads(r.Context())
			if err != nil {
				writeLine(w, map[string]any{"status": "error", "error": err.Error()})
				flusher.Flush()
				return
			}
			for _, task := range tasks {
				if task.ModelName != model.HFRepo {
					continue
				}
				writeLine(w, map[string]any{
					"status":    task.Status,
					"total":     task.TotalSize,
					"completed": task.DownloadedSize,
				})
				flusher.Flush()
				switch task.Status {
				case "completed":
					if err := compat.CheckLocalPath(model.ModelPath); err != nil {
						writeLine(w, map[string]any{"status": "error", "error": err.Error()})
						flusher.Flush()
						return
					}
					writeLine(w, map[string]any{"status": "success"})
					flusher.Flush()
					return
				case "error", "cancelled":
					writeLine(w, map[string]any{"status": "error", "error": "download " + task.Status})
					flusher.Flush()
					return
				}
			}
		}
	}
}

func (s *Server) show(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	model, ok := s.resolve(w, firstNonEmpty(req.Name, req.Model))
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"modelfile":  "FROM " + model.HFRepo,
		"parameters": "",
		"template":   "",
		"details":    model.Details,
		"model_info": map[string]any{
			"openarc.engine":       model.Engine,
			"openarc.model_type":   model.ModelType,
			"openarc.device":       model.Device,
			"huggingface.repo":     model.HFRepo,
			"huggingface.revision": model.HFRevision,
		},
	})
}

func (s *Server) delete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	model, ok := s.resolve(w, firstNonEmpty(req.Name, req.Model))
	if !ok {
		return
	}
	_ = s.manager.UnloadNow(r.Context(), model)
	if err := s.openarc.DeleteModel(r.Context(), model.ModelPath); err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	if !decodeJSON(w, r, &req) {
		return
	}
	if isEmptyMessages(req["messages"]) && isKeepAliveZero(req["keep_alive"]) {
		model, ok := s.resolve(w, stringField(req, "model"))
		if ok {
			if err := s.manager.UnloadNow(r.Context(), model); err != nil {
				writeError(w, http.StatusBadGateway, err)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"model": model.Name, "done": true})
		}
		return
	}
	s.infer(w, r, req, "/v1/chat/completions", "chat")
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	if !decodeJSON(w, r, &req) {
		return
	}
	if prompt, _ := req["prompt"].(string); prompt == "" && isKeepAliveZero(req["keep_alive"]) {
		model, ok := s.resolve(w, stringField(req, "model"))
		if ok {
			if err := s.manager.UnloadNow(r.Context(), model); err != nil {
				writeError(w, http.StatusBadGateway, err)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"model": model.Name, "done": true})
		}
		return
	}
	s.infer(w, r, req, "/v1/completions", "generate")
}

func (s *Server) embed(w http.ResponseWriter, r *http.Request) {
	s.embeddings(w, r)
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	if !decodeJSON(w, r, &req) {
		return
	}
	model, lease, ok := s.prepareInference(w, r.Context(), req)
	if !ok {
		return
	}
	defer lease.Done(r.Context())
	req["model"] = model.Name
	if input, exists := req["prompt"]; exists {
		req["input"] = input
		delete(req, "prompt")
	}
	out, err := s.openarc.ProxyJSON(r.Context(), "/v1/embeddings", req)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) copyModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Source      string `json:"source"`
		Destination string `json:"destination"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	src, ok := s.resolve(w, req.Source)
	if !ok {
		return
	}
	if _, exists := s.manifest.Resolve(req.Destination); exists {
		writeJSON(w, http.StatusOK, map[string]any{"status": "already exists", "source": src.Name, "destination": req.Destination})
		return
	}
	writeError(w, http.StatusNotImplemented, errors.New("copy requires adding the destination alias to models.yaml"))
}

func (s *Server) infer(w http.ResponseWriter, r *http.Request, req map[string]any, path, mode string) {
	model, lease, ok := s.prepareInference(w, r.Context(), req)
	if !ok {
		return
	}
	defer lease.Done(r.Context())
	req["model"] = model.Name
	translateOptions(req)
	stream, _ := req["stream"].(bool)
	if stream {
		s.streamInference(w, r, req, path, model.Name, mode)
		return
	}
	out, err := s.openarc.ProxyJSON(r.Context(), path, req)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, toOllamaResponse(out, model.Name, mode))
}

func (s *Server) prepareInference(w http.ResponseWriter, ctx context.Context, req map[string]any) (*config.Model, *lifecycle.Lease, bool) {
	model, ok := s.resolve(w, stringField(req, "model"))
	if !ok {
		return nil, nil, false
	}
	if !compat.LocalPathLooksPresent(model.ModelPath) {
		if err := s.hf.CheckOpenVINO(ctx, model.HFRepo, model.HFRevision); err != nil {
			writeError(w, http.StatusBadRequest, fmt.Errorf("pre-download OpenVINO check failed for %s: %w", model.HFRepo, err))
			return nil, nil, false
		}
	}
	keepAlive, err := parseKeepAlive(req["keep_alive"])
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return nil, nil, false
	}
	lease, err := s.manager.EnsureLoaded(ctx, model, keepAlive)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return nil, nil, false
	}
	return model, lease, true
}

func (s *Server) streamInference(w http.ResponseWriter, r *http.Request, req map[string]any, path, modelName, mode string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, errors.New("streaming not supported"))
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	err := s.openarc.StreamSSE(r.Context(), path, req, func(chunk map[string]any) error {
		payload := streamChunkToOllama(chunk, modelName, mode)
		if payload == nil {
			return nil
		}
		writeLine(w, payload)
		flusher.Flush()
		return nil
	})
	if err != nil {
		writeLine(w, map[string]any{"model": modelName, "done": true, "error": err.Error()})
		flusher.Flush()
		return
	}
	writeLine(w, map[string]any{"model": modelName, "created_at": time.Now().UTC().Format(time.RFC3339Nano), "done": true})
	flusher.Flush()
}

func (s *Server) resolve(w http.ResponseWriter, name string) (*config.Model, bool) {
	model, ok := s.manifest.Resolve(name)
	if !ok {
		writeError(w, http.StatusNotFound, fmt.Errorf("model %q is not in models.yaml", name))
		return nil, false
	}
	return model, true
}

func tagModel(model config.Model) map[string]any {
	details := model.Details
	if details == nil {
		details = map[string]any{}
	}
	return map[string]any{
		"name":        model.Name,
		"model":       model.Name,
		"modified_at": model.ModifiedAt,
		"size":        model.Size,
		"digest":      model.Digest,
		"details":     details,
	}
}

func translateOptions(req map[string]any) {
	options, _ := req["options"].(map[string]any)
	if options == nil {
		return
	}
	mapping := map[string]string{
		"num_predict":    "max_tokens",
		"temperature":    "temperature",
		"top_p":          "top_p",
		"top_k":          "top_k",
		"repeat_penalty": "repetition_penalty",
	}
	for from, to := range mapping {
		if val, ok := options[from]; ok {
			req[to] = val
		}
	}
	delete(req, "options")
}

func toOllamaResponse(out map[string]any, modelName, mode string) map[string]any {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if mode == "chat" {
		content := ""
		if choices, ok := out["choices"].([]any); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]any); ok {
				if msg, ok := choice["message"].(map[string]any); ok {
					content, _ = msg["content"].(string)
				}
			}
		}
		return map[string]any{"model": modelName, "created_at": now, "message": map[string]any{"role": "assistant", "content": content}, "done": true}
	}
	text := ""
	if choices, ok := out["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			text, _ = choice["text"].(string)
		}
	}
	return map[string]any{"model": modelName, "created_at": now, "response": text, "done": true}
}

func streamChunkToOllama(chunk map[string]any, modelName, mode string) map[string]any {
	choices, ok := chunk["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if mode == "chat" {
		content := ""
		if delta, ok := choice["delta"].(map[string]any); ok {
			content, _ = delta["content"].(string)
		}
		done := choice["finish_reason"] != nil
		return map[string]any{"model": modelName, "created_at": now, "message": map[string]any{"role": "assistant", "content": content}, "done": done}
	}
	text, _ := choice["text"].(string)
	done := choice["finish_reason"] != nil
	return map[string]any{"model": modelName, "created_at": now, "response": text, "done": done}
}

func parseKeepAlive(value any) (*time.Duration, error) {
	if value == nil {
		return nil, nil
	}
	switch v := value.(type) {
	case string:
		d, err := config.ParseDuration(v)
		return &d, err
	case float64:
		d := time.Duration(v * float64(time.Second))
		return &d, nil
	default:
		return nil, fmt.Errorf("unsupported keep_alive value %T", value)
	}
}

func isKeepAliveZero(value any) bool {
	d, err := parseKeepAlive(value)
	return err == nil && d != nil && *d == 0
}

func isEmptyMessages(value any) bool {
	if value == nil {
		return true
	}
	messages, ok := value.([]any)
	return ok && len(messages) == 0
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeLine(w http.ResponseWriter, payload any) {
	bytes, _ := json.Marshal(payload)
	_, _ = w.Write(append(bytes, '\n'))
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]any{"error": err.Error()})
}

func unsupported(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, errors.New("endpoint is not supported by the OpenArc Ollama facade"))
}

func stringField(m map[string]any, key string) string {
	val, _ := m[key].(string)
	return val
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

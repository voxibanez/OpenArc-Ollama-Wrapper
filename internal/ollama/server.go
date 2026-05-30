package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/pprof"
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
	manifest        *config.Manifest
	manager         *lifecycle.Manager
	openarc         *openarc.Client
	hf              *huggingface.Client
	maxRequestBytes int64
}

func NewServer(manifest *config.Manifest, manager *lifecycle.Manager, openArc *openarc.Client, hf *huggingface.Client, maxRequestBytes int64) *Server {
	if maxRequestBytes <= 0 {
		maxRequestBytes = 16 << 20
	}
	return &Server{manifest: manifest, manager: manager, openarc: openArc, hf: hf, maxRequestBytes: maxRequestBytes}
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(s.limitRequestBody)
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
	r.HandleFunc("/debug/pprof/", pprof.Index)
	r.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	r.HandleFunc("/debug/pprof/profile", pprof.Profile)
	r.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	r.HandleFunc("/debug/pprof/trace", pprof.Trace)
	r.Handle("/debug/pprof/allocs", pprof.Handler("allocs"))
	r.Handle("/debug/pprof/goroutine", pprof.Handler("goroutine"))
	r.Handle("/debug/pprof/heap", pprof.Handler("heap"))
	r.Handle("/debug/pprof/threadcreate", pprof.Handler("threadcreate"))
	return r
}

func (s *Server) limitRequestBody(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, s.maxRequestBytes)
		next.ServeHTTP(w, r)
	})
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
	translateThinking(req, mode)
	var err error
	path, err = translateImages(req, model, path, mode)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
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
	normalizer := newReasoningNormalizer(modelName)
	err := s.openarc.StreamSSE(r.Context(), path, req, func(chunk map[string]any) error {
		payload := streamChunkToOllama(chunk, modelName, mode, normalizer)
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

func translateThinking(req map[string]any, mode string) {
	if mode != "chat" {
		delete(req, "think")
		return
	}
	think, ok := req["think"]
	if !ok {
		return
	}
	enabled := false
	switch val := think.(type) {
	case bool:
		enabled = val
	case string:
		enabled = val != "" && val != "false"
	}
	templateKwargs, _ := req["chat_template_kwargs"].(map[string]any)
	if templateKwargs == nil {
		templateKwargs = map[string]any{}
		req["chat_template_kwargs"] = templateKwargs
	}
	if _, exists := templateKwargs["enable_thinking"]; !exists {
		templateKwargs["enable_thinking"] = enabled
	}
	delete(req, "think")
}

func translateImages(req map[string]any, model *config.Model, path, mode string) (string, error) {
	switch mode {
	case "chat":
		hasImages, err := translateChatImages(req)
		if err != nil {
			return path, err
		}
		if hasImages && !isVisionModel(model) {
			return path, fmt.Errorf("model %q is not configured as a vision model in models.yaml", model.Name)
		}
	case "generate":
		images := stringList(req["images"])
		if len(images) == 0 {
			return path, nil
		}
		if !isVisionModel(model) {
			return path, fmt.Errorf("model %q is not configured as a vision model in models.yaml", model.Name)
		}
		req["messages"] = ollamaGenerateMessages(req, images)
		delete(req, "prompt")
		delete(req, "system")
		delete(req, "suffix")
		delete(req, "images")
		delete(req, "raw")
		return "/v1/chat/completions", nil
	}
	return path, nil
}

func translateChatImages(req map[string]any) (bool, error) {
	messages, ok := req["messages"].([]any)
	if !ok {
		return false, nil
	}
	hasImages := false
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			continue
		}
		images := stringList(message["images"])
		if len(images) == 0 {
			continue
		}
		hasImages = true
		message["content"] = openAIContentParts(message["content"], images)
		delete(message, "images")
	}
	return hasImages, nil
}

func ollamaGenerateMessages(req map[string]any, images []string) []any {
	messages := make([]any, 0, 2)
	if system, _ := req["system"].(string); system != "" {
		messages = append(messages, map[string]any{"role": "system", "content": system})
	}
	prompt, _ := req["prompt"].(string)
	messages = append(messages, map[string]any{
		"role":    "user",
		"content": openAIContentParts(prompt, images),
	})
	return messages
}

func openAIContentParts(content any, images []string) []any {
	parts := make([]any, 0, 1+len(images))
	switch value := content.(type) {
	case string:
		if value != "" {
			parts = append(parts, map[string]any{"type": "text", "text": value})
		}
	case []any:
		parts = append(parts, value...)
	case nil:
	default:
		parts = append(parts, map[string]any{"type": "text", "text": fmt.Sprint(value)})
	}
	for _, image := range images {
		parts = append(parts, map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": imageDataURL(image)},
		})
	}
	return parts
}

func imageDataURL(image string) string {
	if strings.HasPrefix(image, "data:image/") {
		return image
	}
	return "data:image/png;base64," + image
}

func stringList(value any) []string {
	switch items := value.(type) {
	case []any:
		out := make([]string, 0, len(items))
		for _, item := range items {
			if text, ok := item.(string); ok && text != "" {
				out = append(out, text)
			}
		}
		return out
	case []string:
		return items
	default:
		return nil
	}
}

func isVisionModel(model *config.Model) bool {
	return strings.EqualFold(model.ModelType, "vlm")
}

func sanitizeGeneratedText(text string) string {
	return strings.ReplaceAll(text, "\uFFFD", "")
}

func toOllamaResponse(out map[string]any, modelName, mode string) map[string]any {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	normalizer := newReasoningNormalizer(modelName)
	if mode == "chat" {
		content := ""
		thinking := ""
		if choices, ok := out["choices"].([]any); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]any); ok {
				if msg, ok := choice["message"].(map[string]any); ok {
					content, _ = msg["content"].(string)
					thinking, _ = msg["reasoning_content"].(string)
					if thinking == "" {
						thinking, _ = msg["thinking"].(string)
					}
				}
			}
		}
		content = sanitizeGeneratedText(content)
		thinking = sanitizeGeneratedText(thinking)
		parts := normalizer.Process(content)
		flush := normalizer.Flush()
		parts.Thinking += flush.Thinking
		parts.Content += flush.Content
		msg := map[string]any{"role": "assistant", "content": parts.Content}
		if thinking != "" || parts.Thinking != "" {
			msg["thinking"] = thinking + parts.Thinking
		}
		return map[string]any{"model": modelName, "created_at": now, "message": msg, "done": true}
	}
	text := ""
	thinking := ""
	if choices, ok := out["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			text, _ = choice["text"].(string)
			if text == "" {
				if msg, ok := choice["message"].(map[string]any); ok {
					text, _ = msg["content"].(string)
				}
			}
			thinking, _ = choice["reasoning_content"].(string)
			if thinking == "" {
				thinking, _ = choice["thinking"].(string)
			}
			if thinking == "" {
				if msg, ok := choice["message"].(map[string]any); ok {
					thinking, _ = msg["reasoning_content"].(string)
					if thinking == "" {
						thinking, _ = msg["thinking"].(string)
					}
				}
			}
		}
		text = sanitizeGeneratedText(text)
		thinking = sanitizeGeneratedText(thinking)
	}
	parts := normalizer.Process(text)
	flush := normalizer.Flush()
	parts.Thinking += flush.Thinking
	parts.Content += flush.Content
	resp := map[string]any{"model": modelName, "created_at": now, "response": parts.Content, "done": true}
	if thinking != "" || parts.Thinking != "" {
		resp["thinking"] = thinking + parts.Thinking
	}
	return resp
}

func streamChunkToOllama(chunk map[string]any, modelName, mode string, normalizer *reasoningNormalizer) map[string]any {
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
		thinking := ""
		if delta, ok := choice["delta"].(map[string]any); ok {
			content, _ = delta["content"].(string)
			thinking, _ = delta["reasoning_content"].(string)
			if thinking == "" {
				thinking, _ = delta["thinking"].(string)
			}
		}
		content = sanitizeGeneratedText(content)
		thinking = sanitizeGeneratedText(thinking)
		done := choice["finish_reason"] != nil
		parts := normalizer.Process(content)
		if done {
			flush := normalizer.Flush()
			parts.Thinking += flush.Thinking
			parts.Content += flush.Content
		}
		msg := map[string]any{"role": "assistant", "content": parts.Content}
		if thinking != "" || parts.Thinking != "" {
			msg["thinking"] = thinking + parts.Thinking
		}
		if msg["content"] == "" && msg["thinking"] == nil && !done {
			return nil
		}
		return map[string]any{"model": modelName, "created_at": now, "message": msg, "done": done}
	}
	text, _ := choice["text"].(string)
	if text == "" {
		if delta, ok := choice["delta"].(map[string]any); ok {
			text, _ = delta["content"].(string)
		}
	}
	thinking, _ := choice["reasoning_content"].(string)
	if thinking == "" {
		thinking, _ = choice["thinking"].(string)
	}
	if thinking == "" {
		if delta, ok := choice["delta"].(map[string]any); ok {
			thinking, _ = delta["reasoning_content"].(string)
			if thinking == "" {
				thinking, _ = delta["thinking"].(string)
			}
		}
	}
	text = sanitizeGeneratedText(text)
	thinking = sanitizeGeneratedText(thinking)
	done := choice["finish_reason"] != nil
	parts := normalizer.Process(text)
	if done {
		flush := normalizer.Flush()
		parts.Thinking += flush.Thinking
		parts.Content += flush.Content
	}
	payload := map[string]any{"model": modelName, "created_at": now, "response": parts.Content, "done": done}
	if thinking != "" || parts.Thinking != "" {
		payload["thinking"] = thinking + parts.Thinking
	}
	if payload["response"] == "" && payload["thinking"] == nil && !done {
		return nil
	}
	return payload
}

type reasoningParts struct {
	Thinking string
	Content  string
}

type reasoningNormalizer struct {
	reasoningModel bool
	probing        bool
	probe          string
	inThink        bool
	tag            string
}

func newReasoningNormalizer(modelName string) *reasoningNormalizer {
	reasoning := isReasoningModel(modelName)
	return &reasoningNormalizer{reasoningModel: reasoning, probing: reasoning}
}

func isReasoningModel(modelName string) bool {
	name := strings.ToLower(modelName)
	return strings.Contains(name, "deepseek-r1") ||
		strings.Contains(name, "qwen3") ||
		strings.Contains(name, "reason") ||
		strings.Contains(name, "thinking")
}

func (n *reasoningNormalizer) Process(text string) reasoningParts {
	if text == "" {
		return reasoningParts{}
	}
	if n.probing {
		n.probe += text
		lower := strings.ToLower(n.probe)
		if openAt := strings.Index(lower, "<think"); openAt >= 0 {
			if openEnd := strings.Index(lower[openAt:], ">"); openEnd >= 0 {
				before := n.probe[:openAt]
				after := n.probe[openAt+openEnd+1:]
				n.probe = ""
				n.probing = false
				n.inThink = true
				parts := n.processTagged(after)
				parts.Content = before + parts.Content
				return parts
			}
		}
		if closeAt := strings.Index(lower, "</think>"); closeAt >= 0 {
			thinking := n.probe[:closeAt]
			after := n.probe[closeAt+len("</think>"):]
			n.probe = ""
			n.probing = false
			n.inThink = false
			parts := n.processTagged(after)
			parts.Thinking = thinking + parts.Thinking
			return parts
		}
		if len(n.probe) < 8192 {
			return reasoningParts{}
		}
		text = n.probe
		n.probe = ""
		n.probing = false
	}
	return n.processTagged(text)
}

func (n *reasoningNormalizer) Flush() reasoningParts {
	var parts reasoningParts
	if n.probe != "" {
		parts.Content = n.probe
		n.probe = ""
		n.probing = false
	}
	if n.tag != "" {
		n.writeText(&parts, n.tag)
		n.tag = ""
	}
	return parts
}

func (n *reasoningNormalizer) processTagged(text string) reasoningParts {
	var parts reasoningParts
	for _, r := range text {
		if n.tag != "" {
			n.tag += string(r)
			if strings.Contains(n.tag, ">") {
				tag := strings.ToLower(n.tag)
				switch {
				case strings.HasPrefix(tag, "<think"):
					n.inThink = true
				case strings.HasPrefix(tag, "</think"):
					n.inThink = false
				default:
					n.writeText(&parts, n.tag)
				}
				n.tag = ""
				continue
			}
			if isThinkTagPrefix(n.tag) {
				continue
			}
			n.writeText(&parts, n.tag)
			n.tag = ""
			continue
		}
		if r == '<' {
			n.tag = "<"
			continue
		}
		n.writeRune(&parts, r)
	}
	return parts
}

func (n *reasoningNormalizer) writeText(parts *reasoningParts, text string) {
	if n.inThink {
		parts.Thinking += text
		return
	}
	parts.Content += text
}

func (n *reasoningNormalizer) writeRune(parts *reasoningParts, r rune) {
	if n.inThink {
		parts.Thinking += string(r)
		return
	}
	parts.Content += string(r)
}

func isThinkTagPrefix(tag string) bool {
	lower := strings.ToLower(tag)
	return strings.HasPrefix("<think", lower) || strings.HasPrefix("</think", lower)
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

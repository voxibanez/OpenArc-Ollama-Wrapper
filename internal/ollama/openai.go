package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/lifecycle"
)

const openAIKeepAliveHeader = "X-OpenArc-Keep-Alive"

func (s *Server) openAIModels(w http.ResponseWriter, r *http.Request) {
	models := make([]map[string]any, 0, len(s.manifest.Models))
	for _, model := range s.manifest.Models {
		models = append(models, openAIModelObject(model))
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   models,
	})
}

func (s *Server) openAIModel(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "*")
	model, ok := s.manifest.Resolve(name)
	if !ok {
		writeOpenAIError(w, http.StatusNotFound, fmt.Errorf("model %q is not in models.yaml", name))
		return
	}
	writeJSON(w, http.StatusOK, openAIModelObject(*model))
}

func openAIModelObject(model config.Model) map[string]any {
	created := int64(0)
	if modified, err := time.Parse(time.RFC3339, model.ModifiedAt); err == nil {
		created = modified.Unix()
	}
	return map[string]any{
		"id":       model.Name,
		"object":   "model",
		"created":  created,
		"owned_by": "openarc",
	}
}

func (s *Server) openAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeOpenAIRequest(w, r)
	if !ok {
		return
	}
	model, lease, ok := s.prepareOpenAIInference(w, r, req)
	if !ok {
		return
	}
	defer lease.Done(context.WithoutCancel(r.Context()))

	req["model"] = model.Name
	translateOptions(req)
	translateThinking(req, "chat")
	path, err := translateImages(req, model, "/v1/chat/completions", "chat")
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err)
		return
	}
	s.proxyOpenAIInference(w, r, req, path)
}

func (s *Server) openAICompletions(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeOpenAIRequest(w, r)
	if !ok {
		return
	}
	model, lease, ok := s.prepareOpenAIInference(w, r, req)
	if !ok {
		return
	}
	defer lease.Done(context.WithoutCancel(r.Context()))

	req["model"] = model.Name
	translateOptions(req)
	translateThinking(req, "generate")
	s.proxyOpenAIInference(w, r, req, "/v1/completions")
}

func (s *Server) openAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeOpenAIRequest(w, r)
	if !ok {
		return
	}
	model, lease, ok := s.prepareOpenAIInference(w, r, req)
	if !ok {
		return
	}
	defer lease.Done(context.WithoutCancel(r.Context()))

	req["model"] = model.Name
	out, err := s.openarc.ProxyJSON(r.Context(), "/v1/embeddings", req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) prepareOpenAIInference(
	w http.ResponseWriter,
	r *http.Request,
	req map[string]any,
) (*config.Model, *lifecycle.Lease, bool) {
	if _, exists := req["keep_alive"]; !exists {
		if keepAlive := r.Header.Get(openAIKeepAliveHeader); keepAlive != "" {
			req["keep_alive"] = keepAlive
		}
	}
	model, lease, status, err := s.acquireInference(r.Context(), req)
	if err != nil {
		writeOpenAIError(w, status, err)
		return nil, nil, false
	}
	delete(req, "keep_alive")
	return model, lease, true
}

func (s *Server) proxyOpenAIInference(
	w http.ResponseWriter,
	r *http.Request,
	req map[string]any,
	path string,
) {
	stream, _ := req["stream"].(bool)
	if stream {
		s.streamOpenAIInference(w, r, req, path)
		return
	}
	out, err := s.openarc.ProxyJSON(r.Context(), path, req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) streamOpenAIInference(
	w http.ResponseWriter,
	r *http.Request,
	req map[string]any,
	path string,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, errors.New("streaming not supported"))
		return
	}

	started := false
	err := s.openarc.StreamSSEWithStart(
		r.Context(),
		path,
		req,
		func() {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			flusher.Flush()
			started = true
		},
		func(chunk map[string]any) error {
			return writeOpenAIEvent(w, flusher, chunk)
		},
	)
	if err != nil {
		if !started {
			writeOpenAIError(w, http.StatusBadGateway, err)
			return
		}
		if r.Context().Err() != nil {
			return
		}
		_ = writeOpenAIEvent(w, flusher, openAIErrorPayload(err, "server_error", nil))
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAIEvent(w http.ResponseWriter, flusher http.Flusher, payload map[string]any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func decodeOpenAIRequest(w http.ResponseWriter, r *http.Request) (map[string]any, bool) {
	defer r.Body.Close()
	var req map[string]any
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err)
		return nil, false
	}
	return req, true
}

func writeOpenAIError(w http.ResponseWriter, status int, err error) {
	errorType := "invalid_request_error"
	var code any
	if status >= http.StatusInternalServerError {
		errorType = "server_error"
	}
	if status == http.StatusNotFound {
		code = "model_not_found"
	}
	writeJSON(w, status, openAIErrorPayload(err, errorType, code))
}

func openAIErrorPayload(err error, errorType string, code any) map[string]any {
	return map[string]any{
		"error": map[string]any{
			"message": err.Error(),
			"type":    errorType,
			"param":   nil,
			"code":    code,
		},
	}
}

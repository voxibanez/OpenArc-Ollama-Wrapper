# OpenArc Ollama Wrapper

Go sidecar that exposes Ollama-style `/api/*` endpoints and delegates model
download, load, inference, and unload to [OpenArc](https://github.com/SearchSavior/OpenArc).

Models are manifest-only. Add every user-visible model to `models.yaml`; unknown
model names return `404` and are never guessed.

## Run

```sh
cp .env.example .env
docker compose up -d
```

The Ollama-compatible endpoint listens on `http://localhost:11434`.

The default compose file pulls both images from GitHub Container Registry:

- `ghcr.io/voxibanez/openarc:latest`
- `ghcr.io/voxibanez/openarc-ollama-wrapper:latest`

The `openarc` image is built from upstream `SearchSavior/OpenArc` with the NPU
driver source build disabled. CPU and Intel GPU runtime packages are still
installed; NPU support should use a custom OpenArc image.

The OpenArc image is large because it includes OpenVINO, OpenVINO GenAI,
Optimum, PyTorch CPU wheels, and Intel GPU runtime packages. Slow pulls usually
stall on the dependency layer; they are normally bandwidth-bound rather than a
facade failure.

For local wrapper development, build the image from this checkout:

```sh
docker compose -f docker-compose.yml -f compose.local-build.yml up --build
```

## Memory diagnostics

The facade is configured with conservative Go runtime limits by default:

- `OLLAMA_FACADE_GOMEMLIMIT=128MiB`
- `OLLAMA_FACADE_GOGC=50`
- `MAX_REQUEST_BYTES=16777216`
- `MAX_RESPONSE_BYTES=67108864`
- `MAX_STREAM_LINE_BYTES=2097152`

If the facade grows unexpectedly, capture profiles before it is killed:

```sh
curl -o heap.pb.gz http://localhost:11434/debug/pprof/heap
curl -o goroutine.txt http://localhost:11434/debug/pprof/goroutine?debug=2
```

## Manifest

Each entry maps an Ollama-facing name to a Hugging Face repo containing
OpenVINO IR files:

```yaml
models:
  - name: qwen3-1.7b
    aliases: [qwen3:1.7b]
    hf_repo: Echo9Zulu/Qwen3-1.7B-int8_asym-ov
    engine: ovgenai
    model_type: llm
    device: GPU.0
```

Before `/api/pull` starts a download, the wrapper checks Hugging Face metadata
for at least one `*_model.xml` and one `*_model.bin`. After download, it checks
the local model path again before OpenArc loads the model.

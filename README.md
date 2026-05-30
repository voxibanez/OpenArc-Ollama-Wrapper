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

The default compose file pulls the wrapper image from GitHub Container Registry:
`ghcr.io/voxibanez/openarc-ollama-wrapper:latest`.

For local wrapper development, build the image from this checkout:

```sh
docker compose -f docker-compose.yml -f compose.local-build.yml up --build
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

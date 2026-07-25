# OpenArc Ollama Wrapper

Go sidecar that exposes Ollama-style `/api/*` and OpenAI-compatible `/v1/*`
endpoints and delegates model download, load, inference, and unload to
[OpenArc](https://github.com/SearchSavior/OpenArc).

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

## OpenAI-compatible API

Point OpenAI clients at `http://localhost:11434/v1`. The facade supports:

- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

`/v1/models` returns every model declared in `models.yaml`, including models
that have not been downloaded or loaded yet. Inference requests use the same
lifecycle manager as the Ollama API: the facade downloads missing models,
reuses an already loaded model, evicts an idle model when the configured limit
is reached, and unloads models after `DEFAULT_KEEP_ALIVE`.

Example:

```sh
curl http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:1.7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

Many OpenAI SDKs require an API key value even when the target does not
authenticate it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-used",
)

response = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The default idle lifetime comes from `DEFAULT_KEEP_ALIVE`. For a per-request
override, send `X-OpenArc-Keep-Alive` with a duration such as `10m`, or `0` to
unload immediately after the request or stream completes. A top-level
`keep_alive` request extension is also accepted and removed before forwarding
the request to OpenArc.

## Custom OpenArc Image

This repo publishes a custom `ghcr.io/voxibanez/openarc:latest` image for
compose deployments. It is still built from upstream
`SearchSavior/OpenArc`, but `Dockerfile.openarc` carries a few deployment
patches while upstream container publishing is still settling:

- Skips the upstream `intel/linux-npu-driver` source build. That path depends on
  transient Launchpad snapshot downloads and has failed in GitHub Actions. CPU
  and Intel GPU runtime packages are still installed; NPU support should use a
  custom OpenArc image.
- Installs Intel Arc GPU userspace packages:
  `intel-opencl-icd`, `intel-level-zero-gpu`, `level-zero`, and
  `level-zero-dev`.
- Keeps `gcc`/`build-essential` in the build image because OpenArc's Python
  dependency tree can compile native extensions, including `evdev` through
  `pynput`.
- Installs current OpenVINO runtime pieces with `uv`, including
  `optimum-intel[openvino]`, `openvino-genai`, and `openvino-tokenizers`.
- Applies `patches/openarc-tokenizer-fallback.py` during the build. This lets
  OpenArc VLM loading handle OpenVINO exports whose local tokenizer metadata
  uses `tokenizer_class=TokenizersBackend` by falling back to the source
  tokenizer declared in `openvino_config.json`.
- Applies `patches/openarc-reasoning-content.py` during the build. This makes
  OpenArc's OpenAI-compatible chat route emit reasoning as
  `reasoning_content` instead of raw `content` when models output
  `<think>...</think>` or the known missing-opening-tag form ending in
  `</think>`.
- Applies `patches/openarc-streaming-unicode.py` during the build. This holds
  back unstable streaming tokenizer deltas that contain the Unicode replacement
  character `U+FFFD` and drops any replacement characters that never stabilize.
- Cleans package and `uv` caches to avoid publishing unnecessary GHCR layer
  weight.

The OpenArc image is large because it includes OpenVINO, OpenVINO GenAI,
Optimum, PyTorch CPU wheels, and Intel GPU runtime packages. Slow pulls usually
stall on the dependency layer; they are normally bandwidth-bound rather than a
facade failure.

For local wrapper development, build the image from this checkout:

```sh
docker compose -f docker-compose.yml -f compose.local-build.yml up --build
```

## Native Proxmox LXC installation

Docker is not required inside an LXC. The native installer runs OpenArc and the
Go facade as separate systemd services and exposes only the Ollama-compatible
API on port `11434`.

The installer currently targets an `amd64` Ubuntu 24.04 LXC. Give it at least
12 GB RAM, with 16 GB preferred for larger models and VLM image processing.

First, on the Proxmox host, identify the Intel Arc render node:

```sh
ls -l /dev/dri
lspci -nnk | grep -A3 -Ei 'VGA|Display'
```

Inside the LXC, get the numeric `render` group ID:

```sh
getent group render
```

Stop the LXC and pass the Arc render node through. Replace `120`, `renderD129`,
and `104` with the actual container ID, device, and in-container render GID:

```sh
pct stop 120
pct set 120 -dev0 path=/dev/dri/renderD129,gid=104,mode=0660
pct start 120
```

PCI/VFIO passthrough is not needed for an LXC. The installer must see a
`/dev/dri/renderD*` character device before it starts.

Clone this repository inside the LXC and run:

```sh
git clone https://github.com/voxibanez/OpenArc-Ollama-Wrapper.git
cd OpenArc-Ollama-Wrapper
sudo ./deploy/lxc/install.sh
```

By default, the installer copies the repository's `models.yaml` only when
`/etc/openarc/models.yaml` does not already exist. To install a private
manifest from another path:

```sh
sudo env \
  MANIFEST_SOURCE=/root/models.yaml \
  OPENARC_API_KEY='replace-with-a-secret' \
  HUGGINGFACE_TOKEN='optional-private-repo-token' \
  ./deploy/lxc/install.sh
```

Set `REPLACE_MANIFEST=1` to deliberately replace an existing installed
manifest. Existing `/etc/openarc/openarc.env` configuration is otherwise
preserved on reruns.

### Updating a native LXC installation

Pull the wrapper repository and run the dedicated updater:

```sh
cd OpenArc-Ollama-Wrapper
git pull --ff-only
sudo ./deploy/lxc/update.sh
```

The updater fetches the requested OpenArc revision, applies this repository's
compatibility patches, creates a new Python environment, rebuilds the Go
facade, and switches the services only after the new release is prepared. It
preserves `/etc/openarc/openarc.env`, `/etc/openarc/models.yaml`, and all
downloaded models.

The first update migrates an installer-created `/opt/openarc` directory into
versioned storage under `/opt/openarc-releases`. If OpenArc or the facade does
not pass its local health check after activation, the updater restores the
previous OpenArc release and facade binary. By default it keeps the active
release and one previous release.

To update to a branch, tag, or commit other than `main`, or retain more
rollback versions:

```sh
sudo env OPENARC_REF=v2.0.5 KEEP_RELEASES=3 ./deploy/lxc/update.sh
```

The updater does not update its own wrapper checkout. Run `git pull --ff-only`
first so the facade and OpenArc patches are built from the desired wrapper
revision.

The native installation uses:

- `/opt/openarc` for the active patched OpenArc checkout and Python environment.
- `/opt/openarc-releases` for versioned OpenArc releases and rollback.
- `/usr/local/bin/ollama-openarc` for the compiled Go facade.
- `/var/lib/openarc/models` for downloaded models.
- `/etc/openarc/models.yaml` for the private model manifest.
- `/etc/openarc/openarc.env` for service configuration and secrets.

Verify the installation:

```sh
systemctl status openarc ollama-openarc
curl http://127.0.0.1:11434/api/version
curl http://127.0.0.1:11434/api/tags
journalctl -u openarc -u ollama-openarc -f
```

The installer applies the same tokenizer, reasoning-content, and streaming
Unicode patches used by the custom OpenArc container image.

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

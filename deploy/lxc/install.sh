#!/usr/bin/env bash
set -Eeuo pipefail

OPENARC_REF="${OPENARC_REF:-main}"
OPENARC_DIR="${OPENARC_DIR:-/opt/openarc}"
STATE_DIR="${STATE_DIR:-/var/lib/openarc}"
MODEL_DIR="${MODEL_DIR:-${STATE_DIR}/models}"
CONFIG_DIR="${CONFIG_DIR:-/etc/openarc}"
MANIFEST_DEST="${MANIFEST_DEST:-${CONFIG_DIR}/models.yaml}"
ENV_FILE="${ENV_FILE:-${CONFIG_DIR}/openarc.env}"
SERVICE_USER="${SERVICE_USER:-openarc}"
GO_VERSION="${GO_VERSION:-1.24.4}"
REPLACE_MANIFEST="${REPLACE_MANIFEST:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MANIFEST_SOURCE="${MANIFEST_SOURCE:-${REPO_ROOT}/models.yaml}"
GPU_DEVICES=()
GPU_GROUPS=()
SERVICE_GROUP="${SERVICE_USER}"

log() {
  printf '[openarc-lxc] %s\n' "$*"
}

fail() {
  printf '[openarc-lxc] ERROR: %s\n' "$*" >&2
  exit 1
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || fail "run this installer as root"
  [[ "$(uname -m)" == "x86_64" ]] || fail "this Intel GPU installer currently supports x86_64 only"
  [[ -d /run/systemd/system ]] || fail "systemd is required inside the LXC"
}

check_os() {
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || \
    fail "Ubuntu 24.04 is required; found ${PRETTY_NAME:-unknown OS}"
}

discover_gpu_access() {
  local render_devices=(/dev/dri/renderD*)
  [[ -e "${render_devices[0]}" ]] || fail \
    "no /dev/dri/renderD* device is visible; pass the Arc render node into the LXC from the Proxmox host first"
  RENDER_DEVICE="${RENDER_DEVICE:-${render_devices[0]}}"
  [[ -c "${RENDER_DEVICE}" ]] || fail "${RENDER_DEVICE} is not a character device"

  local devices=(/dev/dri/renderD* /dev/dri/card*)
  local device gid group
  local -A seen_devices=()
  local -A seen_groups=()
  for device in "${devices[@]}" "${RENDER_DEVICE}"; do
    [[ -c "${device}" && -z "${seen_devices[${device}]:-}" ]] || continue
    seen_devices["${device}"]=1
    GPU_DEVICES+=("${device}")

    gid="$(stat -c '%g' "${device}")"
    group="$(getent group "${gid}" | cut -d: -f1 || true)"
    if [[ -z "${group}" ]]; then
      group="openarc-gpu-${gid}"
      if getent group "${group}" >/dev/null; then
        [[ "$(getent group "${group}" | cut -d: -f3)" == "${gid}" ]] || \
          fail "group ${group} already exists with the wrong gid"
      else
        groupadd --gid "${gid}" "${group}"
      fi
    fi
    if [[ -z "${seen_groups[${group}]:-}" ]]; then
      seen_groups["${group}"]=1
      GPU_GROUPS+=("${group}")
    fi
    log "using GPU device ${device} (group ${group}, gid ${gid})"
  done

  for group in render video; do
    if getent group "${group}" >/dev/null && [[ -z "${seen_groups[${group}]:-}" ]]; then
      seen_groups["${group}"]=1
      GPU_GROUPS+=("${group}")
    fi
  done
}

install_system_packages() {
  log "installing system dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clinfo \
    curl \
    gcc \
    git \
    gpg \
    gpg-agent \
    libasound2-dev \
    libgl1 \
    libglib2.0-0 \
    libudev-dev \
    ocl-icd-libopencl1 \
    openssl \
    portaudio19-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget

  wget -qO- https://repositories.intel.com/gpu/intel-graphics.key |
    gpg --dearmor --yes --output /usr/share/keyrings/intel-graphics.gpg
  cat >/etc/apt/sources.list.d/intel-gpu-noble.list <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client
EOF
  apt-get update
  apt-get install -y --no-install-recommends \
    intel-level-zero-gpu \
    intel-opencl-icd \
    level-zero \
    level-zero-dev
}

install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
  fi
}

install_go() {
  local installed=""
  if command -v go >/dev/null 2>&1; then
    installed="$(go env GOVERSION 2>/dev/null || true)"
  fi
  installed="${installed#go}"
  if [[ -n "${installed}" ]] && dpkg --compare-versions "${installed}" ge "${GO_VERSION}"; then
    return
  fi

  log "installing Go ${GO_VERSION}"
  local archive
  archive="$(mktemp)"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o "${archive}"
  tar -tzf "${archive}" >/dev/null
  rm -rf /usr/local/go
  tar -C /usr/local -xzf "${archive}"
  rm -f "${archive}"
  ln -sf /usr/local/go/bin/go /usr/local/bin/go
  ln -sf /usr/local/go/bin/gofmt /usr/local/bin/gofmt
}

create_service_user() {
  if ! getent group "${SERVICE_GROUP}" >/dev/null; then
    groupadd --system "${SERVICE_GROUP}"
  fi
  if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
    useradd \
      --system \
      --gid "${SERVICE_GROUP}" \
      --home-dir "${STATE_DIR}" \
      --create-home \
      --shell /usr/sbin/nologin \
      "${SERVICE_USER}"
  else
    usermod --gid "${SERVICE_GROUP}" --home "${STATE_DIR}" --shell /usr/sbin/nologin "${SERVICE_USER}"
  fi

  local supplemental_groups
  supplemental_groups="$(IFS=,; printf '%s' "${GPU_GROUPS[*]}")"
  [[ -z "${supplemental_groups}" ]] || usermod --append --groups "${supplemental_groups}" "${SERVICE_USER}"

  install -d -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" -m 0750 \
    "${STATE_DIR}" "${MODEL_DIR}" "${STATE_DIR}/cache" "${STATE_DIR}/huggingface"
  chown "${SERVICE_USER}:${SERVICE_GROUP}" \
    "${STATE_DIR}" "${MODEL_DIR}" "${STATE_DIR}/cache" "${STATE_DIR}/huggingface"
  install -d -o root -g "${SERVICE_GROUP}" -m 0750 "${CONFIG_DIR}"

  local device
  for device in "${GPU_DEVICES[@]}"; do
    if ! runuser -u "${SERVICE_USER}" -- test -r "${device}" ||
       ! runuser -u "${SERVICE_USER}" -- test -w "${device}"; then
      fail "${SERVICE_USER} cannot read and write ${device}; correct the LXC device gid/mode on the Proxmox host"
    fi
  done
}

install_openarc() {
  if [[ ! -d "${OPENARC_DIR}/.git" ]]; then
    [[ ! -e "${OPENARC_DIR}" ]] || fail "${OPENARC_DIR} exists but is not an OpenArc git checkout"
    log "cloning OpenArc (${OPENARC_REF})"
    git clone --branch "${OPENARC_REF}" --depth 1 \
      https://github.com/SearchSavior/OpenArc.git "${OPENARC_DIR}"
  else
    log "using existing OpenArc checkout at ${OPENARC_DIR}"
    log "set OPENARC_DIR to a new path to install a different OpenArc revision"
  fi

  local openarc_target
  openarc_target="$(readlink -f "${OPENARC_DIR}")"
  chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${openarc_target}"

  if ! grep -q 'def load_vlm_tokenizer' "${OPENARC_DIR}/src/engine/ov_genai/vlm.py"; then
    runuser -u "${SERVICE_USER}" -- python3 - \
      "${OPENARC_DIR}/src/engine/ov_genai/vlm.py" \
      <"${REPO_ROOT}/patches/openarc-tokenizer-fallback.py"
  fi
  if ! grep -q 'class ReasoningSplitter' "${OPENARC_DIR}/src/server/routes/openai.py"; then
    runuser -u "${SERVICE_USER}" -- python3 - \
      "${OPENARC_DIR}/src/server/routes/openai.py" \
      <"${REPO_ROOT}/patches/openarc-reasoning-content.py"
  fi
  if ! grep -q 'def _emit_decoded' "${OPENARC_DIR}/src/engine/ov_genai/streamers.py"; then
    runuser -u "${SERVICE_USER}" -- python3 - \
      "${OPENARC_DIR}/src/engine/ov_genai/streamers.py" \
      <"${REPO_ROOT}/patches/openarc-streaming-unicode.py"
  fi

  log "installing OpenArc Python dependencies"
  (
    cd "${OPENARC_DIR}"
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv sync
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv pip install --pre -U openvino-genai openvino-tokenizers \
      --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" \
      "${OPENARC_DIR}/.venv/bin/python" -m compileall -q \
      "${OPENARC_DIR}/src" "${OPENARC_DIR}/.venv/lib/python3.12/site-packages"
  )

  ln -sfn "${STATE_DIR}/openarc_config.json" "${OPENARC_DIR}/openarc_config.json"
  chown -h "${SERVICE_USER}:${SERVICE_GROUP}" "${OPENARC_DIR}/openarc_config.json"
}

install_facade() {
  log "building the Ollama facade"
  (
    cd "${REPO_ROOT}"
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
      go build -trimpath -ldflags='-s -w' -o /usr/local/bin/ollama-openarc ./cmd/ollama-openarc
  )
  chown root:root /usr/local/bin/ollama-openarc
  chmod 0755 /usr/local/bin/ollama-openarc
}

install_configuration() {
  [[ -f "${MANIFEST_SOURCE}" ]] || fail "manifest source not found: ${MANIFEST_SOURCE}"
  if [[ ! -f "${MANIFEST_DEST}" || "${REPLACE_MANIFEST}" == "1" ]]; then
    log "installing model manifest at ${MANIFEST_DEST}"
    install -o root -g "${SERVICE_USER}" -m 0640 "${MANIFEST_SOURCE}" "${MANIFEST_DEST}"
  else
    log "preserving existing manifest at ${MANIFEST_DEST}"
  fi

  if [[ ! -f "${ENV_FILE}" ]]; then
    local api_key
    api_key="${OPENARC_API_KEY:-$(openssl rand -hex 32)}"
    umask 0027
    cat >"${ENV_FILE}" <<EOF
OPENARC_API_KEY=${api_key}
OPENARC_MODELS_DIR=${MODEL_DIR}
OPENARC_BASE_URL=http://127.0.0.1:8000
MODEL_PATH=${MODEL_DIR}
MODEL_MANIFEST=${MANIFEST_DEST}
LISTEN_ADDR=0.0.0.0:11434
OPENARC_DEFAULT_DEVICE=GPU.0
MAX_LOADED_MODELS=1
DEFAULT_KEEP_ALIVE=60s
IDLE_CHECK_INTERVAL=10s
DOWNLOAD_POLL_INTERVAL=1s
GOMEMLIMIT=128MiB
GOGC=50
MAX_REQUEST_BYTES=16777216
MAX_RESPONSE_BYTES=67108864
MAX_STREAM_LINE_BYTES=2097152
HF_HOME=${STATE_DIR}/huggingface
NEOReadDebugKeys=1
OverrideGpuAddressSpace=48
EnableImplicitScaling=1
EOF
    if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
      printf 'HUGGINGFACE_TOKEN=%s\n' "${HUGGINGFACE_TOKEN}" >>"${ENV_FILE}"
    fi
    chown root:"${SERVICE_USER}" "${ENV_FILE}"
    chmod 0640 "${ENV_FILE}"
  else
    log "preserving existing environment file at ${ENV_FILE}"
  fi
}

install_systemd_units() {
  log "installing systemd services"
  cat >/etc/systemd/system/openarc.service <<EOF
[Unit]
Description=OpenArc inference server
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${OPENARC_DIR}
EnvironmentFile=${ENV_FILE}
Environment=HOME=${STATE_DIR}
Environment=PATH=${OPENARC_DIR}/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=${OPENARC_DIR}/.venv/bin/openarc serve start --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=${STATE_DIR}
UMask=0027

[Install]
WantedBy=multi-user.target
EOF

  cat >/etc/systemd/system/ollama-openarc.service <<EOF
[Unit]
Description=Ollama-compatible facade for OpenArc
Requires=openarc.service
After=openarc.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
EnvironmentFile=${ENV_FILE}
Environment=HOME=${STATE_DIR}
ExecStart=/usr/local/bin/ollama-openarc
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=${STATE_DIR}
UMask=0027

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable openarc.service ollama-openarc.service
  systemctl restart openarc.service

  local attempts=0
  until (
    # EnvironmentFile syntax is intentionally shell-compatible.
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    curl -fsS \
      -H "Authorization: Bearer ${OPENARC_API_KEY:-}" \
      http://127.0.0.1:8000/v1/models >/dev/null
  ); do
    attempts=$((attempts + 1))
    if (( attempts >= 60 )); then
      systemctl --no-pager --full status openarc.service || true
      journalctl --no-pager -n 100 -u openarc.service || true
      fail "OpenArc did not become ready"
    fi
    sleep 1
  done

  systemctl restart ollama-openarc.service
  attempts=0
  until curl -fsS http://127.0.0.1:11434/api/version >/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts >= 30 )); then
      systemctl --no-pager --full status ollama-openarc.service || true
      journalctl --no-pager -n 100 -u ollama-openarc.service || true
      fail "Ollama facade did not become ready"
    fi
    sleep 1
  done
}

verify_gpu() {
  log "checking OpenVINO device discovery"
  runuser -u "${SERVICE_USER}" -- env \
    HOME="${STATE_DIR}" \
    "${OPENARC_DIR}/.venv/bin/python" -c \
    'from openvino import Core; devices = Core().available_devices; print("OpenVINO devices:", devices); assert any(d.startswith("GPU") for d in devices), "no GPU device found"'
}

main() {
  require_root
  exec 9>/run/lock/openarc-lxc-update.lock
  flock -n 9 || fail "another OpenArc install or update is running"

  check_os
  discover_gpu_access
  install_system_packages
  install_uv
  install_go
  create_service_user
  install_openarc
  install_facade
  install_configuration
  install_systemd_units
  verify_gpu

  log "installation complete"
  log "Ollama API: http://$(hostname -I | awk '{print $1}'):11434"
  log "Manifest: ${MANIFEST_DEST}"
  log "Configuration: ${ENV_FILE}"
  log "Logs: journalctl -u openarc -u ollama-openarc -f"
}

main "$@"

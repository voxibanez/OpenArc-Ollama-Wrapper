#!/usr/bin/env bash
set -Eeuo pipefail

OPENARC_REF="${OPENARC_REF:-main}"
OPENARC_REPOSITORY="${OPENARC_REPOSITORY:-https://github.com/SearchSavior/OpenArc.git}"
OPENARC_DIR="${OPENARC_DIR:-/opt/openarc}"
RELEASE_ROOT="${RELEASE_ROOT:-/opt/openarc-releases}"
STATE_DIR="${STATE_DIR:-/var/lib/openarc}"
MODEL_DIR="${MODEL_DIR:-${STATE_DIR}/models}"
CONFIG_DIR="${CONFIG_DIR:-/etc/openarc}"
ENV_FILE="${ENV_FILE:-${CONFIG_DIR}/openarc.env}"
SERVICE_USER="${SERVICE_USER:-openarc}"
KEEP_RELEASES="${KEEP_RELEASES:-2}"
RUNTIME_REVISION=1
SERVICE_GROUP="${SERVICE_USER}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STAGING_DIR=""
NEW_RELEASE=""
NEW_RELEASE_CREATED=0
NEW_FACADE=""
FACADE_BACKUP=""
PREVIOUS_KIND=""
PREVIOUS_TARGET=""
LEGACY_RELEASE=""
SWITCHED=0
GPU_DEVICES=()
GPU_GROUPS=()

log() {
  printf '[openarc-lxc-update] %s\n' "$*"
}

fail() {
  printf '[openarc-lxc-update] ERROR: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  [[ -z "${STAGING_DIR}" || ! -d "${STAGING_DIR}" ]] || rm -rf "${STAGING_DIR}"
  if (( NEW_RELEASE_CREATED == 1 )) && [[ -n "${NEW_RELEASE}" && -d "${NEW_RELEASE}" ]]; then
    rm -rf "${NEW_RELEASE}"
  fi
  [[ -z "${NEW_FACADE}" || ! -e "${NEW_FACADE}" ]] || rm -f "${NEW_FACADE}"
  [[ -z "${FACADE_BACKUP}" || ! -e "${FACADE_BACKUP}" ]] || rm -f "${FACADE_BACKUP}"
}
trap cleanup EXIT

check_prerequisites() {
  [[ "${EUID}" -eq 0 ]] || fail "run this updater as root"
  [[ "$(uname -m)" == "x86_64" ]] || fail "this Intel GPU updater currently supports x86_64 only"
  [[ -d /run/systemd/system ]] || fail "systemd is required"
  [[ -f "${ENV_FILE}" ]] || fail "missing ${ENV_FILE}; run deploy/lxc/install.sh first"
  [[ -e "${OPENARC_DIR}" ]] || fail "missing ${OPENARC_DIR}; run deploy/lxc/install.sh first"
  local required
  for required in curl flock git go python3 runuser sha256sum systemctl uv; do
    command -v "${required}" >/dev/null 2>&1 || fail "required command not found: ${required}"
  done

  [[ "${KEEP_RELEASES}" =~ ^[1-9][0-9]*$ ]] || fail "KEEP_RELEASES must be a positive integer"
  [[ "${OPENARC_DIR}" == /* && "${OPENARC_DIR}" != "/" ]] || fail "OPENARC_DIR must be a non-root absolute path"
  [[ "${RELEASE_ROOT}" == /* && "${RELEASE_ROOT}" != "/" ]] || fail "RELEASE_ROOT must be a non-root absolute path"
  [[ "${OPENARC_DIR}" != "${RELEASE_ROOT}" ]] || fail "OPENARC_DIR and RELEASE_ROOT must be different"
  [[ -f "${REPO_ROOT}/patches/openarc-tokenizer-fallback.py" ]] || fail "run this script from a complete wrapper checkout"
  [[ -f /etc/systemd/system/openarc.service ]] || fail "openarc.service is not installed"
  [[ -f /etc/systemd/system/ollama-openarc.service ]] || fail "ollama-openarc.service is not installed"
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

ensure_service_user() {
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

  local device
  for device in "${GPU_DEVICES[@]}"; do
    if ! runuser -u "${SERVICE_USER}" -- test -r "${device}" ||
       ! runuser -u "${SERVICE_USER}" -- test -w "${device}"; then
      fail "${SERVICE_USER} cannot read and write ${device}; correct the LXC device gid/mode on the Proxmox host"
    fi
  done

  local current_openarc
  current_openarc="$(readlink -f "${OPENARC_DIR}")"
  [[ ! -d "${current_openarc}" ]] || chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${current_openarc}"
}

apply_patches() {
  local source_dir="$1"

  python3 "${REPO_ROOT}/patches/openarc-tokenizer-fallback.py" \
    "${source_dir}/src/engine/ov_genai/vlm.py"
  python3 "${REPO_ROOT}/patches/openarc-reasoning-content.py" \
    "${source_dir}/src/server/routes/openai.py"
  python3 "${REPO_ROOT}/patches/openarc-streaming-unicode.py" \
    "${source_dir}/src/engine/ov_genai/streamers.py"
}

prepare_openarc_release() {
  install -d -o root -g root -m 0755 "${RELEASE_ROOT}"
  STAGING_DIR="$(mktemp -d "${RELEASE_ROOT}/.staging.XXXXXX")"

  log "fetching OpenArc revision ${OPENARC_REF}"
  git -C "${STAGING_DIR}" init -q
  git -C "${STAGING_DIR}" remote add origin "${OPENARC_REPOSITORY}"
  git -C "${STAGING_DIR}" fetch -q --depth 1 origin "${OPENARC_REF}"
  git -C "${STAGING_DIR}" checkout -q --detach FETCH_HEAD

  local commit runtime_digest release_id
  commit="$(git -C "${STAGING_DIR}" rev-parse HEAD)"
  runtime_digest="$(
    {
      sha256sum \
        "${REPO_ROOT}/patches/openarc-tokenizer-fallback.py" \
        "${REPO_ROOT}/patches/openarc-reasoning-content.py" \
        "${REPO_ROOT}/patches/openarc-streaming-unicode.py"
      printf 'runtime-revision %s\n' "${RUNTIME_REVISION}"
    } |
      sha256sum |
      cut -c1-12
  )"
  release_id="${commit:0:12}-${runtime_digest}"
  NEW_RELEASE="${RELEASE_ROOT}/${release_id}"

  if [[ -f "${NEW_RELEASE}/.openarc-wrapper-release" ]] &&
     grep -qx "${commit} ${runtime_digest}" "${NEW_RELEASE}/.openarc-wrapper-release"; then
    log "reusing prepared OpenArc release ${release_id}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${NEW_RELEASE}"
    rm -rf "${STAGING_DIR}"
    STAGING_DIR=""
    return
  fi
  [[ ! -e "${NEW_RELEASE}" ]] || fail "incomplete release already exists: ${NEW_RELEASE}"

  mv "${STAGING_DIR}" "${NEW_RELEASE}"
  STAGING_DIR=""
  NEW_RELEASE_CREATED=1
  chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${NEW_RELEASE}"
  chmod 0755 "${NEW_RELEASE}"

  log "applying wrapper compatibility patches"
  apply_patches "${NEW_RELEASE}"
  chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${NEW_RELEASE}"

  log "installing OpenArc Python dependencies"
  (
    cd "${NEW_RELEASE}"
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv sync
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" UV_CACHE_DIR="${STATE_DIR}/cache" \
      uv pip install --pre -U openvino-genai openvino-tokenizers \
      --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    runuser -u "${SERVICE_USER}" -- env HOME="${STATE_DIR}" \
      "${NEW_RELEASE}/.venv/bin/python" -m compileall -q \
      "${NEW_RELEASE}/src" "${NEW_RELEASE}/.venv/lib/python3.12/site-packages"
  )

  ln -sfn "${STATE_DIR}/openarc_config.json" "${NEW_RELEASE}/openarc_config.json"
  chown -h "${SERVICE_USER}:${SERVICE_GROUP}" "${NEW_RELEASE}/openarc_config.json"
  runuser -u "${SERVICE_USER}" -- \
    bash -c 'printf "%s %s\n" "$1" "$2" >"$3"' \
    _ "${commit}" "${runtime_digest}" "${NEW_RELEASE}/.openarc-wrapper-release"
  NEW_RELEASE_CREATED=0
  log "prepared OpenArc release ${release_id}"
}

build_facade() {
  log "building the Ollama/OpenAI facade"
  NEW_FACADE="$(mktemp /usr/local/bin/.ollama-openarc.XXXXXX)"
  (
    cd "${REPO_ROOT}"
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
      go build -trimpath -ldflags='-s -w' -o "${NEW_FACADE}" ./cmd/ollama-openarc
  )
  chown root:root "${NEW_FACADE}"
  chmod 0755 "${NEW_FACADE}"
}

verify_staged_gpu() {
  log "checking OpenVINO device discovery in the staged release"
  runuser -u "${SERVICE_USER}" -- env \
    HOME="${STATE_DIR}" \
    "${NEW_RELEASE}/.venv/bin/python" -c \
    'from openvino import Core; devices = Core().available_devices; print("OpenVINO devices:", devices); assert any(d.startswith("GPU") for d in devices), "no GPU device found"'
}

openarc_ready() {
  local attempts=0
  # EnvironmentFile syntax generated by the installer is shell-compatible.
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  until curl -fsS \
    -H "Authorization: Bearer ${OPENARC_API_KEY:-}" \
    http://127.0.0.1:8000/v1/models >/dev/null; do
    attempts=$((attempts + 1))
    (( attempts < 60 )) || return 1
    sleep 1
  done
}

facade_ready() {
  local attempts=0
  until curl -fsS http://127.0.0.1:11434/api/version >/dev/null; do
    attempts=$((attempts + 1))
    (( attempts < 30 )) || return 1
    sleep 1
  done
}

record_previous_installation() {
  if [[ -L "${OPENARC_DIR}" ]]; then
    PREVIOUS_KIND="symlink"
    PREVIOUS_TARGET="$(readlink -f "${OPENARC_DIR}")"
    [[ -d "${PREVIOUS_TARGET}" ]] || fail "${OPENARC_DIR} points to a missing release"
  elif [[ -d "${OPENARC_DIR}" ]]; then
    PREVIOUS_KIND="directory"
    LEGACY_RELEASE="${RELEASE_ROOT}/legacy-$(date -u +%Y%m%d%H%M%S)"
    [[ ! -e "${LEGACY_RELEASE}" ]] || fail "legacy release path already exists: ${LEGACY_RELEASE}"
  else
    fail "${OPENARC_DIR} is not a directory or symlink"
  fi
}

switch_openarc_release() {
  local link_path="${OPENARC_DIR}.new"
  rm -f "${link_path}" || return 1

  if [[ "${PREVIOUS_KIND}" == "directory" ]]; then
    mv "${OPENARC_DIR}" "${LEGACY_RELEASE}" || return 1
  else
    rm "${OPENARC_DIR}" || return 1
  fi
  SWITCHED=1

  ln -s "${NEW_RELEASE}" "${link_path}" || return 1
  mv -T "${link_path}" "${OPENARC_DIR}" || return 1
}

rollback() {
  log "health check failed; rolling back"
  systemctl stop ollama-openarc.service openarc.service || true

  if (( SWITCHED == 1 )); then
    rm -f "${OPENARC_DIR}"
    rm -f "${OPENARC_DIR}.new"
    if [[ "${PREVIOUS_KIND}" == "directory" ]]; then
      mv "${LEGACY_RELEASE}" "${OPENARC_DIR}"
    else
      ln -s "${PREVIOUS_TARGET}" "${OPENARC_DIR}"
    fi
  fi

  if [[ -n "${FACADE_BACKUP}" && -f "${FACADE_BACKUP}" ]]; then
    install -o root -g root -m 0755 "${FACADE_BACKUP}" /usr/local/bin/ollama-openarc
  fi

  systemctl start openarc.service || true
  if openarc_ready; then
    systemctl start ollama-openarc.service || true
  fi
  systemctl --no-pager --full status openarc.service ollama-openarc.service || true
  journalctl --no-pager -n 100 -u openarc.service -u ollama-openarc.service || true
}

activate_release() {
  record_previous_installation
  if [[ -f /usr/local/bin/ollama-openarc ]]; then
    FACADE_BACKUP="$(mktemp /usr/local/bin/.ollama-openarc.backup.XXXXXX)"
    cp -a /usr/local/bin/ollama-openarc "${FACADE_BACKUP}"
  fi

  log "stopping services for release switch"
  systemctl stop ollama-openarc.service openarc.service || return 1
  switch_openarc_release || return 1
  install -o root -g root -m 0755 "${NEW_FACADE}" /usr/local/bin/ollama-openarc || return 1
  rm -f "${NEW_FACADE}"
  NEW_FACADE=""

  systemctl start openarc.service || return 1
  openarc_ready || return 1
  systemctl start ollama-openarc.service || return 1
  facade_ready || return 1
}

prune_releases() {
  local current release
  local -a releases
  current="$(readlink -f "${OPENARC_DIR}")"

  mapfile -t releases < <(
    find "${RELEASE_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name '.staging.*' \
      -printf '%T@ %p\n' |
      sort -rn |
      cut -d' ' -f2-
  )

  local kept=0
  for release in "${releases[@]}"; do
    if [[ "${release}" == "${current}" ]]; then
      continue
    fi
    kept=$((kept + 1))
    if (( kept >= KEEP_RELEASES )); then
      log "removing old release ${release}"
      rm -rf "${release}"
    fi
  done
}

main() {
  check_prerequisites
  exec 9>/run/lock/openarc-lxc-update.lock
  flock -n 9 || fail "another OpenArc install or update is running"

  discover_gpu_access
  ensure_service_user
  prepare_openarc_release
  verify_staged_gpu
  build_facade

  if ! activate_release; then
    rollback
    fail "update failed and the previous release was restored"
  fi

  prune_releases
  log "update complete"
  log "OpenArc release: $(readlink -f "${OPENARC_DIR}")"
  log "Ollama API: http://$(hostname -I | awk '{print $1}'):11434"
}

main "$@"

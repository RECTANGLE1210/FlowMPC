#!/usr/bin/env bash
# Run inside tmux; no git pull, credential file, GPU reset or automatic restart.
# DRY_RUN=1 prints the command without asking for a key, checking resources,
# authenticating, importing CUDA, or starting training.
#
# Optional environment settings (integer resource thresholds):
# TARGET_GPU_UUID         Pin a known GPU UUID instead of discovering index 1.
# MIN_VRAM_MIB=8500       MAX_GPU_UTILIZATION=15   READY_CHECKS=3
# MIN_RAM_GIB=40          MIN_DISK_GIB=5           POLL_SECONDS=60
# NVIDIA_QUERY_TIMEOUT_SECONDS=10  CUDA_TEST_TIMEOUT_SECONDS=30
# WANDB_PREFLIGHT_TIMEOUT_SECONDS=20
# TDMPC2_DIR              Defaults to this script's parent directory.
# DATA_DIR, TD_CHECKPOINT, FM_CHECKPOINT: default server paths below.
# DISK_PATH=/mnt/disk2    CONDA_SH: optional conda.sh installation path.
# MEMINFO_PATH=/proc/meminfo (primarily useful for mocked tests).
#
# The key is private shell memory while waiting, then an environment credential
# only for authentication/online training. Privileged users can still inspect
# memory/environment. The key is lost if this tmux/server process terminates.
# Readiness is a heuristic, not a GPU reservation or an OOM/failure guarantee.
# UUID discovery pins the first accessible index 1; specify TARGET_GPU_UUID for
# identity across reboots/index changes. No query depends on GPU0 being healthy.
# W&B fallback happens BEFORE launch only. A later Python/W&B failure exits;
# the launcher never restarts a partially completed training process.
# Existing Python code owns checkpoint validation, atomic latest.pt saves and
# terminal logs. Eval scheduling retains the existing TD-MPC2 iteration index.

set +x
set -euo pipefail
umask 077
unset WANDB_KEY WANDB_API_KEY
WANDB_KEY=''
TRAIN_PID=''
WAIT_PID=''

log() {
    local line
    line="[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
    printf '%s\n' "$line"
    if [[ -n ${LOG_FD:-} ]]; then
        printf '%s\n' "$line" >&"$LOG_FD"
    fi
}

die() { log "ERROR: $*"; exit 1; }

cleanup() {
    local status=$? pid child_status
    trap - EXIT INT TERM
    unset WANDB_KEY WANDB_API_KEY
    # Terminate only children created by THIS launcher, never other GPU users.
    for pid in "$TRAIN_PID" "$WAIT_PID"; do
        if [[ -n $pid ]]; then
            kill -TERM "$pid" 2>/dev/null || true
            if wait "$pid" 2>/dev/null; then child_status=0; else child_status=$?; fi
            log "Own child PID=$pid exit code=$child_status"
        fi
    done
    log "Launcher exit code=$status"
    if [[ -n ${LOG_FD:-} ]]; then exec {LOG_FD}>&-; fi
}

prompt_key() {
    [[ -r /dev/tty && -w /dev/tty ]] || die 'A controlling terminal is required; run inside tmux.'
    printf 'Enter W&B API key: ' >/dev/tty
    if ! IFS= read -r -s WANDB_KEY </dev/tty; then
        printf '\n' >/dev/tty
        die 'Could not read the key from the terminal.'
    fi
    printf '\n' >/dev/tty
}

configure() {
    TDMPC2_DIR=${TDMPC2_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
    cd -- "$TDMPC2_DIR" || die "Repository directory missing: $TDMPC2_DIR"
    TDMPC2_DIR=$PWD
    DATA_DIR=${DATA_DIR:-/mnt/disk2/pmduy/data/80}
    TD_CHECKPOINT=${TD_CHECKPOINT:-/mnt/disk2/hungnc/checkpoints/mt80-48M.pt}
    FM_CHECKPOINT=${FM_CHECKPOINT:-/mnt/disk2/hungnc/FlowMPC/tdmpc2/logs/mt80/1/mt80_fm_top10/flowmpc/fm/flow_matching_model.pt}
    DISK_PATH=${DISK_PATH:-/mnt/disk2}
    MEMINFO_PATH=${MEMINFO_PATH:-/proc/meminfo}
    MIN_VRAM_MIB=${MIN_VRAM_MIB:-8500}
    MAX_GPU_UTILIZATION=${MAX_GPU_UTILIZATION:-15}
    READY_CHECKS=${READY_CHECKS:-3}
    MIN_RAM_GIB=${MIN_RAM_GIB:-40}
    MIN_DISK_GIB=${MIN_DISK_GIB:-5}
    POLL_SECONDS=${POLL_SECONDS:-60}
    NVIDIA_QUERY_TIMEOUT_SECONDS=${NVIDIA_QUERY_TIMEOUT_SECONDS:-10}
    CUDA_TEST_TIMEOUT_SECONDS=${CUDA_TEST_TIMEOUT_SECONDS:-30}
    WANDB_PREFLIGHT_TIMEOUT_SECONDS=${WANDB_PREFLIGHT_TIMEOUT_SECONDS:-20}
    WORK_DIR="$TDMPC2_DIR/logs/mt80/1/flowmpc_full_12task_512k"
    LATEST="$WORK_DIR/models/latest.pt"
    mkdir -p -- "$WORK_DIR/launcher"
    LAUNCHER_LOG="$WORK_DIR/launcher/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM.log"
    # Exclusive creation, including launches in the same second.
    (set -o noclobber; : >"$LAUNCHER_LOG") || die 'Cannot create launcher log.'
    exec {LOG_FD}>>"$LAUNCHER_LOG"
    log "Launcher started at=${STARTED_AT:-unknown}; log=$LAUNCHER_LOG; dry_run=$DRY_RUN"
    local name value
    for name in MIN_VRAM_MIB MAX_GPU_UTILIZATION READY_CHECKS MIN_RAM_GIB MIN_DISK_GIB POLL_SECONDS NVIDIA_QUERY_TIMEOUT_SECONDS CUDA_TEST_TIMEOUT_SECONDS WANDB_PREFLIGHT_TIMEOUT_SECONDS; do
        value=${!name}
        [[ $value =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 9 ]] || die "$name must be a nonnegative decimal integer."
    done
    (( READY_CHECKS > 0 && POLL_SECONDS > 0 && NVIDIA_QUERY_TIMEOUT_SECONDS > 0 && CUDA_TEST_TIMEOUT_SECONDS > 0 && WANDB_PREFLIGHT_TIMEOUT_SECONDS > 0 && MAX_GPU_UTILIZATION <= 100 )) || die 'Invalid check count, interval, timeout or utilization threshold.'
    [[ -z ${TARGET_GPU_UUID:-} || ${TARGET_GPU_UUID} =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] || die 'TARGET_GPU_UUID must be a complete NVIDIA GPU UUID.'
    GPU_UUID=${TARGET_GPU_UUID:-}
    log "Thresholds: VRAM>=${MIN_VRAM_MIB}MiB utilization<=${MAX_GPU_UTILIZATION}% RAM>=${MIN_RAM_GIB}GiB disk>=${MIN_DISK_GIB}GiB checks=$READY_CHECKS poll=${POLL_SECONDS}s"
}

activate_conda() {
    local base
    if [[ ${CONDA_DEFAULT_ENV:-} != flowmpc ]]; then
        if [[ -z ${CONDA_SH:-} ]]; then
            if [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
                base=$("$CONDA_EXE" info --base) || die 'Cannot locate Conda installation.'
            elif command -v conda >/dev/null 2>&1; then
                base=$(conda info --base) || die 'Cannot locate Conda installation.'
            elif [[ -r $HOME/miniconda3/etc/profile.d/conda.sh ]]; then
                base="$HOME/miniconda3"
            elif [[ -r $HOME/anaconda3/etc/profile.d/conda.sh ]]; then
                base="$HOME/anaconda3"
            else
                die 'Conda not found; initialize Conda or set CONDA_SH.'
            fi
            CONDA_SH="$base/etc/profile.d/conda.sh"
        fi
        [[ -r $CONDA_SH ]] || die "Conda initialization missing: $CONDA_SH"
        # Some Conda activation scripts require nounset to be disabled.
        set +u
        source "$CONDA_SH"
        if ! conda activate flowmpc; then
            set -u
            die 'Cannot activate the flowmpc Conda environment.'
        fi
        set -u
    fi
    [[ ${CONDA_DEFAULT_ENV:-} == flowmpc ]] || die 'flowmpc Conda environment is not active.'
    PYTHON=$(command -v python) || die 'Python is missing from the flowmpc environment.'
    log "Conda environment=flowmpc; Python=$PYTHON"
}

validate_inputs() {
    [[ -r flowmpc/train.py ]] || die 'TDMPC2_DIR does not contain flowmpc/train.py.'
    [[ -d $DATA_DIR && -r $DATA_DIR && -x $DATA_DIR ]] || die "Dataset directory unavailable: $DATA_DIR"
    local file found=0
    for file in "$DATA_DIR"/*.pt; do
        if [[ -f $file && -r $file ]]; then found=1; break; fi
    done
    (( found )) || die "No readable dataset chunks in $DATA_DIR"
    [[ -f $FM_CHECKPOINT && -r $FM_CHECKPOINT ]] || die "FM checkpoint unavailable: $FM_CHECKPOINT"
    if [[ -e $LATEST || -L $LATEST ]]; then
        [[ -f $LATEST && -r $LATEST ]] || die "Existing latest.pt is not a readable file: $LATEST"
    else
        [[ -f $TD_CHECKPOINT && -r $TD_CHECKPOINT ]] || die "Pretrained checkpoint unavailable: $TD_CHECKPOINT"
    fi
    # Preserve installed MuJoCo configuration. Supply only known defaults.
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    if [[ -z ${MUJOCO_PY_MUJOCO_PATH:-} && -d $HOME/.mujoco/mujoco210 ]]; then
        export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
    fi
    if [[ -n ${MUJOCO_PY_MUJOCO_PATH:-} && -d $MUJOCO_PY_MUJOCO_PATH/bin ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$MUJOCO_PY_MUJOCO_PATH/bin:"*) ;;
            *) export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
}

resources_ready() {
    local row uuid bus total free utilization extra gpu_ok=0
    if row=$(timeout --kill-after=5s "${NVIDIA_QUERY_TIMEOUT_SECONDS}s" nvidia-smi --id="${GPU_UUID:-1}" --query-gpu=uuid,pci.bus_id,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null); then
        IFS=, read -r uuid bus total free utilization extra <<<"$row"
        uuid=${uuid//[[:space:]]/}; bus=${bus//[[:space:]]/}
        total=${total//[[:space:]]/}; free=${free//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
        if [[ $row != *$'\n'* && -z $extra && $uuid =~ ^GPU-[[:xdigit:]-]+$ && -n $bus && $total =~ ^[0-9]+$ && $free =~ ^[0-9]+$ && $utilization =~ ^[0-9]+$ ]]; then
            if [[ -z $GPU_UUID ]]; then
                GPU_UUID=$uuid
                log "Pinned physical index 1 to UUID=$GPU_UUID PCI=$bus"
            fi
            if [[ $uuid == "$GPU_UUID" ]]; then
                gpu_ok=1
            fi
        fi
    fi
    RAM_KIB=$(awk '/^MemAvailable:/ {print $2; exit}' "$MEMINFO_PATH" 2>/dev/null) || RAM_KIB=''
    DISK_KIB=$(df -Pk -- "$DISK_PATH" 2>/dev/null | awk 'NR==2 {print $4}') || DISK_KIB=''
    if (( ! gpu_ok )); then
        log "GPU ${GPU_UUID:-physical-index-1} inaccessible; RAM=${RAM_KIB:-unavailable}KiB disk=${DISK_KIB:-unavailable}KiB; waiting"
        return 1
    fi
    log "GPU accessible UUID=$uuid PCI=$bus total=${total}MiB free=${free}MiB utilization=${utilization}% RAM=${RAM_KIB:-unavailable}KiB disk=${DISK_KIB:-unavailable}KiB"
    [[ $RAM_KIB =~ ^[0-9]+$ && $DISK_KIB =~ ^[0-9]+$ ]] || return 1
    (( 10#$free >= MIN_VRAM_MIB && 10#$utilization <= MAX_GPU_UTILIZATION && 10#$RAM_KIB >= MIN_RAM_GIB * 1048576 && 10#$DISK_KIB >= MIN_DISK_GIB * 1048576 ))
}

cuda_ready() {
    # Only the pinned device is visible; physical GPU1 is PyTorch cuda:0.
    CUDA_VISIBLE_DEVICES="$GPU_UUID" CUDA_DEVICE_ORDER=PCI_BUS_ID timeout --kill-after=5s "${CUDA_TEST_TIMEOUT_SECONDS}s" "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.ones((64, 64), device="cuda:0")
y = x @ x
torch.cuda.synchronize()
assert y[0, 0].item() == 64
PY
}

wait_for_resources() {
    local consecutive=0
    while true; do
        if resources_ready; then
            consecutive=$((consecutive + 1))
            log "Resources ready: consecutive=$consecutive/$READY_CHECKS"
            if (( consecutive >= READY_CHECKS )); then
                if cuda_ready; then
                    log "Isolated PyTorch CUDA test passed UUID=$GPU_UUID (visible cuda:0)"
                    return
                fi
                log 'Isolated PyTorch CUDA test failed/timed out; returning to wait'
                consecutive=0
            fi
        else
            consecutive=0
            log 'Resource check unsuccessful; consecutive checks reset'
        fi
        sleep "$POLL_SECONDS" &
        WAIT_PID=$!
        wait "$WAIT_PID"
        WAIT_PID=''
    done
}

authenticate_wandb() {
    set +x  # Also guard against tracing accidentally enabled by Conda hooks.
    export WANDB_MODE=offline
    if [[ -n $WANDB_KEY ]]; then
        export WANDB_API_KEY="$WANDB_KEY"
        # An explicit in-memory API key bypasses wandb.login/netrc writes.
        # Suppress all SDK diagnostics: errors must never expose credentials.
        if WANDB_MODE=online timeout --kill-after=5s "${WANDB_PREFLIGHT_TIMEOUT_SECONDS}s" "$PYTHON" - <<'PY' >/dev/null 2>&1
import os
import wandb
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=5)
assert api.viewer.username
PY
        then
            export WANDB_MODE=online
            log 'W&B preflight succeeded; WANDB_MODE=online'
        else
            unset WANDB_API_KEY
            log 'W&B preflight failed/timed out; WANDB_MODE=offline'
        fi
    else
        log 'No API key provided; WANDB_MODE=offline'
    fi
    unset WANDB_KEY
}

build_command() {
    RESUME=null
    if [[ -e $LATEST || -L $LATEST ]]; then
        [[ -f $LATEST && -r $LATEST ]] || die "Existing latest.pt is not a readable file: $LATEST"
        RESUME=$LATEST
        log "Resume checkpoint=$RESUME; FlowMPC will validate compatibility (never fall back to fresh training)"
    else
        log "Fresh pretrained initialization=$TD_CHECKPOINT"
    fi
    local tasks='[1,3,5,11,18,29,30,41,47,56,63,71]'
    COMMAND=("${PYTHON:-python}" -m flowmpc.train
        task=mt80 obs=state model_size=48 flowmpc_train_mode=full
        "flowmpc_train_task_ids=$tasks" "flowmpc_eval_task_ids=$tasks"
        "data_dir=$DATA_DIR" "tdmpc_checkpoint=$TD_CHECKPOINT" "fm_checkpoint=$FM_CHECKPOINT"
        steps=512000 batch_size=256 eval_freq=50000 eval_episodes=3
        flowmpc_save_freq=10000 "flowmpc_resume_checkpoint=$RESUME"
        seed=1 exp_name=flowmpc_full_12task_512k checkpoint=null
        enable_wandb=true wandb_project=flowmpc-mt80 wandb_entity=nguyenconghung1210 save_agent=true)
    local printable
    printf -v printable '%q ' "${COMMAND[@]}"
    log "Training command (no secrets): $printable"
}

report_resume_count() {
    [[ $RESUME != null ]] || return 0
    local count
    # Read a trusted local training file only for status; validation remains in
    # FlowMPC. Failure here must NOT discard it or switch to fresh initialization.
    if count=$(timeout --kill-after=5s 30s "$PYTHON" - "$RESUME" <<'PY' 2>/dev/null
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
update = state["update"]
assert type(update) is int and update >= 0
print(update)
PY
    ); then
        [[ $count =~ ^[0-9]+$ ]] && log "Checkpoint completed updates=$count; target TOTAL updates=512000"
    else
        log 'Checkpoint count unavailable; Python will validate/report the resume point'
    fi
    return 0
}

main() {
    trap cleanup EXIT
    trap 'log "SIGINT received; stopping this launcher and its own child only"; exit 130' INT
    trap 'log "SIGTERM received; stopping this launcher and its own child only"; exit 143' TERM
    DRY_RUN=${DRY_RUN:-0}
    [[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die 'DRY_RUN must be 0 or 1.'
    STARTED_AT=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $DRY_RUN == 0 ]]; then prompt_key; fi
    configure
    if [[ $DRY_RUN == 1 ]]; then
        build_command
        log 'DRY_RUN: no key requested, resource wait, authentication or training performed'
        return
    fi
    local tool
    for tool in nvidia-smi timeout flock awk df sleep; do
        command -v "$tool" >/dev/null 2>&1 || die "Required command missing: $tool"
    done
    exec {LOCK_FD}>"$WORK_DIR/.launcher.lock"
    flock -n "$LOCK_FD" || die 'Another launcher/training process holds this experiment lock.'
    activate_conda
    validate_inputs
    wait_for_resources
    export CUDA_VISIBLE_DEVICES="$GPU_UUID" CUDA_DEVICE_ORDER=PCI_BUS_ID
    export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
    log "Selected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; readiness is a heuristic, not a reservation"
    build_command
    report_resume_count
    authenticate_wandb
    log "Training started; Python terminal logs: $WORK_DIR/terminal/"
    "${COMMAND[@]}" </dev/null &
    TRAIN_PID=$!
    log "Training PID=$TRAIN_PID"
    local status
    if wait "$TRAIN_PID"; then status=0; else status=$?; fi
    TRAIN_PID=''
    log "Training process exit code=$status; no automatic restart"
    return "$status"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi

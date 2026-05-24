#!/usr/bin/env bash
# =============================================================================
# SabiYarn — GCP TPU VM setup
# =============================================================================
#
# PART 1 runs on your LOCAL machine (anywhere gcloud is installed).
# PART 2 runs INSIDE the TPU VM after you SSH in.
#
# Usage
# -----
# 1. Fill in the variables in the CONFIG block below.
# 2. Run Part 1 locally to create the VM and GCS bucket.
# 3. SSH into the VM and run Part 2 to install deps and start training.
#
# TPU accelerator reference
# -------------------------
#   v3-8    →  8 cores, 128 GB HBM   (cheapest, good for development)
#   v4-8    →  8 cores, 192 GB HBM   (recommended for this model size)
#   v4-32   →  32 cores, 4-host pod  (run Part 2 on all 4 hosts in parallel)
#   v5e-8   →  8 cores, next-gen efficiency
#
# All v*-8 variants are single-host; pod slices (v4-32, etc.) require
# running the training command on every host simultaneously — see the
# pod-slice section at the bottom of Part 2.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIG — edit these before running
# =============================================================================

PROJECT="sabiyarn"
ZONE="us-central1-a"               # zones with TPU v4: us-central2-b, europe-west4-a
TPU_NAME="sabiyarn-tpu"
ACCELERATOR="v4-8"                 # change to v3-8 for cheaper dev runs
TPU_VERSION="tpu-vm-pt-2.1"       # PyTorch 2.1 + torch_xla 2.1 pre-installed
DISK_SIZE="200GB"                  # local SSD for data + checkpoints

GCS_BUCKET="gs://${PROJECT}-sabiyarn-data"   # will be created if missing
REPO_URL="https://github.com/NaijaAI/sabiyarn.git"
BRANCH="distributed"

WANDB_API_KEY=""                   # paste your key or leave blank to skip
HF_TOKEN=""                        # HuggingFace token for dataset access


# =============================================================================
# PART 1 — run on your LOCAL machine
# =============================================================================
# To execute Part 1: bash scripts/gcp_tpu_setup.sh --create
# =============================================================================

part1_create() {
    echo "=== Setting active project ==="
    gcloud config set project "${PROJECT}"

    echo "=== Creating GCS bucket (data + checkpoints) ==="
    # -l must match the region of your TPU zone (e.g. us-central1 for us-central1-a)
    REGION="${ZONE%-*}"   # strips the trailing -a / -b suffix
    gcloud storage buckets create "${GCS_BUCKET}" \
        --location="${REGION}" \
        --uniform-bucket-level-access \
        2>/dev/null || echo "Bucket already exists — skipping"

    echo "=== Creating TPU VM ==="
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
        --zone="${ZONE}" \
        --accelerator-type="${ACCELERATOR}" \
        --version="${TPU_VERSION}" \
        --data-disk="source=projects/${PROJECT}/zones/${ZONE}/disks/${TPU_NAME}-disk,mode=read-write" \
        2>/dev/null || true   # ignore if already exists

    # Create the persistent disk if it doesn't exist yet
    gcloud compute disks create "${TPU_NAME}-disk" \
        --zone="${ZONE}" \
        --size="${DISK_SIZE}" \
        --type="pd-balanced" \
        2>/dev/null || echo "Disk already exists — skipping"

    echo ""
    echo "TPU VM created. SSH into it with:"
    echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
    echo ""
    echo "Then run Part 2 inside the VM:"
    echo "  bash scripts/gcp_tpu_setup.sh --setup"
}

# SSH helper: run a command on all workers of a pod slice
part1_ssh_all() {
    CMD="$1"
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --zone="${ZONE}" \
        --worker=all \
        --command="${CMD}"
}


# =============================================================================
# PART 2 — run INSIDE the TPU VM
# =============================================================================
# Either SSH in and run this file with: bash scripts/gcp_tpu_setup.sh --setup
# Or copy-paste the individual sections as needed.
# =============================================================================

part2_setup() {
    echo "=== [1/6] System packages ==="
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        git \
        gcsfuse \
        tmux \
        htop \
        nvme-cli

    echo "=== [2/6] Mount persistent disk ==="
    sudo mkdir -p /data
    # The disk device name depends on how it was attached; check with `lsblk`
    DISK_DEV=$(lsblk -o NAME,SIZE | grep -E "200G|199G" | awk '{print $1}' | head -1)
    if [[ -n "${DISK_DEV}" ]]; then
        # Format only if not already formatted
        sudo blkid "/dev/${DISK_DEV}" || sudo mkfs.ext4 -F "/dev/${DISK_DEV}"
        sudo mount "/dev/${DISK_DEV}" /data || true
        echo "/dev/${DISK_DEV} /data ext4 defaults 0 2" | sudo tee -a /etc/fstab
        sudo chmod 777 /data
    else
        echo "Persistent disk not found — using ephemeral /data"
        sudo mkdir -p /data
        sudo chmod 777 /data
    fi

    echo "=== [3/6] Python dependencies ==="
    # torch + torch_xla are pre-installed on the TPU VM image.
    # We install only the additional packages the training script needs.
    pip install --quiet \
        transformers \
        wandb \
        structlog \
        "numpy<2.0" \
        sentencepiece \
        omegaconf \
        datasets \
        python-dotenv \
        psutil \
        pyyaml \
        lmdb

    # bitsandbytes is CUDA-only; the training code handles its absence gracefully.
    # gputil is GPU-only; skip it here.

    echo "=== [4/6] Clone / update repo ==="
    if [[ -d "/opt/sabiyarn/.git" ]]; then
        cd /opt/sabiyarn && git fetch origin && git checkout "${BRANCH}" && git pull
    else
        sudo mkdir -p /opt/sabiyarn
        sudo chmod 777 /opt/sabiyarn
        git clone --branch "${BRANCH}" "${REPO_URL}" /opt/sabiyarn
    fi
    cd /opt/sabiyarn

    echo "=== [5/6] Environment variables ==="
    # PJRT_DEVICE=TPU is required for PyTorch/XLA 2.x (replaces the old XRT_TPU_CONFIG)
    cat >> ~/.bashrc << 'EOF'
export PJRT_DEVICE=TPU
export PYTHONPATH=/opt/sabiyarn:${PYTHONPATH:-}
EOF

    # Write secrets to a .env file (not committed to git)
    cat > /opt/sabiyarn/.env << ENVEOF
PJRT_DEVICE=TPU
WANDB_API_KEY=${WANDB_API_KEY}
HF_TOKEN=${HF_TOKEN}
ENVEOF

    source ~/.bashrc

    echo "=== [6/6] Verify torch_xla ==="
    python - << 'PYEOF'
import torch
import torch_xla
import torch_xla.core.xla_model as xm
device = xm.xla_device()
x = torch.ones(3, 3, device=device)
xm.mark_step()
print(f"torch        : {torch.__version__}")
print(f"torch_xla    : {torch_xla.__version__}")
print(f"XLA device   : {device}")
print(f"TPU available: OK")
PYEOF

    echo ""
    echo "Setup complete. See 'Preparing data' section below before training."
}


# =============================================================================
# Data preparation — run inside the TPU VM
# =============================================================================
# Option A (recommended): Store processed bins in GCS, stream to local disk.
# Option B (simple):      Run the tokeniser directly on the TPU VM.
# =============================================================================

prepare_data() {
    source ~/.bashrc
    cd /opt/sabiyarn

    # --- Option A: sync pre-processed data from GCS ---
    # If you have already run prepare.py on a GPU machine and uploaded the bins:
    #   gcloud storage cp gs://<bucket>/train.bin /data/train.bin
    #   gcloud storage cp gs://<bucket>/val.bin   /data/val.bin

    # --- Option B: tokenise here (requires HuggingFace access) ---
    echo "Tokenising dataset (this may take a while)..."
    TRAIN_DATA_PATH=/data/train.bin \
    VAL_DATA_PATH=/data/val.bin \
    python -c "
from data.prepare import run
run(
    datasets=['Aletheia-ng/pretrain_test'],
    hf_repo_files={},   # use all files
    num_proc=96,        # TPU VMs have many CPU cores
    n_samples=-1,
    seed=42,
    hash_algo='md5',
    registry_cache='/data/global_hash_registry.lmdb',
    map_size_gb=50,
)
"
    # Back up processed bins to GCS so you don't need to re-run this
    gcloud storage cp /data/train.bin "${GCS_BUCKET}/train.bin"
    gcloud storage cp /data/val.bin   "${GCS_BUCKET}/val.bin"
    echo "Data ready at /data/train.bin and /data/val.bin"
}


# =============================================================================
# Training — run inside the TPU VM
# =============================================================================

run_training() {
    source ~/.bashrc
    cd /opt/sabiyarn

    export PJRT_DEVICE=TPU
    export PYTHONPATH=/opt/sabiyarn

    # TPU_NUM_DEVICES is read by tpu_launch.py to determine core count.
    # v4-8 / v3-8 / v5e-8 all have 8 cores per host.
    export TPU_NUM_DEVICES=8

    # Run inside screen so the job survives SSH disconnection.
    # Attach later with: screen -r sabiyarn
    screen -dmS sabiyarn bash -c "
        python training/tpu_launch.py \
            --config training/train_config.yaml \
            --num_cores 8 \
            2>&1 | tee /data/train.log
    "
    echo "Training started in background screen session 'sabiyarn'."
    echo "Attach with:  screen -r sabiyarn"
    echo "Monitor logs: tail -f /data/train.log"
}


# =============================================================================
# Pod-slice training (v4-32 and larger)
# =============================================================================
# For multi-host pod slices, run the training command on ALL hosts at once.
# From your LOCAL machine (not inside the VM):
#
#   gcloud compute tpus tpu-vm ssh sabiyarn-tpu \
#       --zone=us-central1-a \
#       --worker=all \
#       --command="cd /opt/sabiyarn && PJRT_DEVICE=TPU python training/tpu_launch.py --num_cores 8"
#
# The XLA runtime (via TPU_WORKER_HOSTNAMES and TPU_WORKER_ID set by GCP)
# coordinates across hosts automatically — no MASTER_ADDR needed.
# =============================================================================


# =============================================================================
# Checkpoint sync to GCS — run periodically or as a cron job
# =============================================================================

sync_checkpoints() {
    gcloud storage rsync /data/checkpoints "${GCS_BUCKET}/checkpoints" --recursive
    echo "Checkpoints synced to ${GCS_BUCKET}/checkpoints"
}


# =============================================================================
# Teardown — run on your LOCAL machine when done
# =============================================================================

teardown() {
    echo "Stopping TPU VM (preserves disk and GCS data)..."
    gcloud compute tpus tpu-vm stop "${TPU_NAME}" --zone="${ZONE}"

    echo "To delete the VM entirely (disk and GCS data are kept):"
    echo "  gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}"
}


# =============================================================================
# Entrypoint
# =============================================================================

case "${1:-}" in
    --create)  part1_create   ;;
    --setup)   part2_setup    ;;
    --data)    prepare_data   ;;
    --train)   run_training   ;;
    --sync)    sync_checkpoints ;;
    --stop)    teardown        ;;
    *)
        echo "Usage: $0 [--create | --setup | --data | --train | --sync | --stop]"
        echo ""
        echo "  --create   (local)  Create TPU VM and GCS bucket"
        echo "  --setup    (on VM)  Install deps, mount disk, verify XLA"
        echo "  --data     (on VM)  Tokenise dataset and upload to GCS"
        echo "  --train    (on VM)  Start training in a screen session"
        echo "  --sync     (on VM)  Sync checkpoints to GCS"
        echo "  --stop     (local)  Stop the TPU VM"
        ;;
esac

#!/usr/bin/env python3
"""
TPU training launcher for SabiYarn.

Spawns one process per TPU core via torch_xla.distributed.xla_multiprocessing.
Each worker initialises torch.distributed with backend='xla', so every existing
dist.all_reduce / dist.all_gather call in MLA and MoE works with zero changes.

Usage on a TPU VM
-----------------
Single host (v3-8, v4-8, v5e-8 — 8 cores):
    python training/tpu_launch.py

Single host, explicit core count:
    python training/tpu_launch.py --num_cores 8

Custom config:
    python training/tpu_launch.py --config /path/to/config.yaml

Pod slice (run this same command on EVERY host in the pod):
    python training/tpu_launch.py --num_cores 8

Pod-slice coordination is handled automatically by the XLA runtime via
the TPU_WORKER_HOSTNAMES / TPU_WORKER_ID environment variables that GCP
sets on each TPU VM. You do not need a MASTER_ADDR or MASTER_PORT.

TPU core counts by accelerator type
------------------------------------
v3-8, v4-8, v5e-8   →  8 cores  (single host, run once)
v4-32, v5e-32        → 32 cores  (4-host pod slice, run on each of the 4 hosts)
v4-128, v5e-256      → ...        (larger pod slices, same pattern)

Environment variables (set automatically on GCP TPU VMs, override locally)
---------------------------------------------------------------------------
TPU_NUM_DEVICES     : number of TPU cores on this host  (default: 8)
PJRT_DEVICE         : 'TPU' for hardware, 'CPU' for unit tests
"""

import argparse
import os
import sys

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import yaml
import structlog
import torch.distributed as dist

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend  # noqa: F401 — registers 'xla' dist backend

from sabiyarn.model import AttentionType
from training.new_train import TrainingConfig
from training.xla_trainer import XLASabiYarnTrainer

LOG = structlog.stdlib.get_logger()


# ---------------------------------------------------------------------------
# Per-core worker
# ---------------------------------------------------------------------------

def worker_fn(local_rank: int, config: TrainingConfig) -> None:
    """
    Entry point executed once per TPU core.

    local_rank is set automatically by xmp.spawn (0..nprocs-1 on this host).
    Global rank (across all hosts in a pod) comes from dist after init.

    The 'xla://' init_method uses XLA's built-in rendezvous — no TCP store,
    no MASTER_ADDR needed. Works for both single-host and pod-slice jobs.
    """
    dist.init_process_group(backend="xla", init_method="xla://")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = xm.xla_device()

    # Patch config fields that are per-process or XLA-specific.
    # We do this inside the worker (after spawn) so the mutations don't
    # affect the original config object in the parent process.
    config.dist = True
    config.world_size = world_size
    # TrainingConfig.device is a str; xm.xla_device() returns a torch.device.
    config.device = str(device)          # e.g. "xla:0"
    config.dtype = "bfloat16"           # TPUs support bfloat16 natively; no scaler needed
    config.compile_model = False         # XLA is the compiler; torch.compile is a no-op here
    config.dist_strategy = "xla-ddp"    # signals SabiYarnTrainer to use XLA-aware DDP
    config.backend = "xla"

    # Divide gradient accumulation steps across cores so that the effective
    # batch size stays constant regardless of world_size.
    if config.gradient_accumulation_steps % world_size != 0:
        raise ValueError(
            f"gradient_accumulation_steps ({config.gradient_accumulation_steps}) "
            f"must be divisible by world_size ({world_size})"
        )
    config.gradient_accumulation_steps //= world_size

    if global_rank == 0:
        LOG.info(
            "XLA distributed training started",
            world_size=world_size,
            device=str(device),
        )

    trainer = XLASabiYarnTrainer(config)
    trainer.train()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> TrainingConfig:
    """
    Build a TrainingConfig from a YAML file, with XLA-appropriate overrides.

    Fields like device/dtype/compile_model are overridden here to safe defaults
    and then further patched per-process inside worker_fn.
    """
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    return TrainingConfig(
        # --- Architecture ---
        attention_type=AttentionType(conf["model"]["attention_type"]),
        dim=conf["model"]["dim"],
        n_layers=conf["model"]["n_layers"],
        n_heads=conf["model"]["n_heads"],
        n_kv_heads=conf["model"]["n_kv_heads"],
        vocab_size=conf["model"]["vocab_size"],
        max_seq_len=conf["model"]["max_seq_len"],
        max_batch_size=conf["training"]["max_batch_size"],
        tie_weights=conf["model"]["tie_weights"],
        norm_eps=conf["model"].get("norm_eps", 1e-5),
        init_std=conf["model"].get("init_std", 0.02),
        use_j_linear=conf["model"].get("use_j_linear", True),
        use_logic_network=conf["model"].get("use_logic_network", False),
        # --- MoE ---
        use_moe=conf["model"]["use_moe"],
        n_routed_experts=conf["model"]["n_routed_experts"],
        n_activated_experts=conf["model"]["n_activated_experts"],
        moe_inter_dim=conf["model"]["moe_inter_dim"],
        n_shared_experts=conf["model"]["n_shared_experts"],
        score_function=conf["model"]["score_function"],
        bias_update_speed=conf["model"]["bias_update_speed"],
        moe_aux_loss_weight=conf["model"]["moe_aux_loss_weight"],
        # --- Multi-Token Prediction ---
        use_multi_token_prediction=conf["model"]["use_multi_token_prediction"],
        num_prediction_tokens=conf["model"]["num_prediction_tokens"],
        mtp_only_training=conf["model"]["mtp_only_training"],
        # --- Layer sharing ---
        layer_sharing=conf["model"]["layer_sharing"],
        layer_sharing_strategy=conf["model"]["layer_sharing_strategy"],
        n_unique_layers=conf["model"]["n_unique_layers"],
        # --- Loss ---
        use_cut_cross_entropy=conf["model"]["use_cut_cross_entropy"],
        # --- Training ---
        train_batch_size=conf["training"]["train_batch_size"],
        gradient_accumulation_steps=conf["training"]["gradient_accumulation_steps"],
        learning_rate=conf["training"]["learning_rate"],
        max_iters=conf["training"]["max_iters"],
        weight_decay=conf["training"]["weight_decay"],
        warmup_iters=conf["training"]["warmup_iters"],
        lr_decay_iters=conf["training"]["lr_decay_iters"],
        grad_clip=conf["training"]["grad_clip"],
        optimizer_type=conf["training"]["optimizer_type"],
        use_custom_causal_mask=conf["training"]["use_custom_causal_mask"],
        enable_generation_during_training=conf["training"]["enable_generation_during_training"],
        # --- Data ---
        dataset=conf["data"]["datasets"],
        train_data_path=conf["data"]["train_data_path"],
        eval_data_path=conf["data"]["eval_data_path"],
        out_dir=conf["data"]["out_dir"],
        overwrite_data=conf["data"]["overwrite_data"],
        # --- Logging ---
        eval_interval=conf["wandb"]["eval_interval"],
        log_interval=conf["wandb"]["log_interval"],
        wandb_log=conf["wandb"]["log"],
        wandb_project=conf["wandb"]["project"],
        wandb_run_name=conf["wandb"]["wandb_run_name"],
        wandb_tags=["TPU", "XLA", conf["model"]["attention_type"], "SabiYarn"],
        log_grad_norm=conf["training"]["log_grad_norm"],
        # --- Hashing / dedup ---
        hash_algo=conf["hash"]["hash_algo"],
        registry_cache=conf["hash"]["registry_cache"],
        map_size_gb=conf["hash"]["map_size_gb"],
        # --- Init ---
        init_from=conf["training"]["init_from"],
        # --- XLA overrides (will be patched further in worker_fn) ---
        device="xla",
        dtype="bfloat16",
        compile_model=False,
        dist=True,
        dist_strategy="xla-ddp",
        backend="xla",
        auto_detect_distributed=False,  # XLA handles this; skip the CUDA detection path
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SabiYarn TPU training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "train_config.yaml"),
        help="Path to training config YAML (default: training/train_config.yaml)",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help=(
            "Number of TPU cores to use on this host. "
            "Defaults to $TPU_NUM_DEVICES or 8."
        ),
    )
    args = parser.parse_args()

    num_cores = args.num_cores or int(os.environ.get("TPU_NUM_DEVICES", "8"))

    LOG.info("Loading config", path=args.config)
    config = load_config(args.config)

    LOG.info(
        "Launching SabiYarn TPU training",
        num_cores=num_cores,
        attention=config.attention_type,
        dim=config.dim,
        layers=config.n_layers,
    )

    # xmp.spawn forks `num_cores` child processes, passing local_rank as the
    # first argument to worker_fn, then the contents of `args`.
    # The parent process blocks here until all workers finish or one crashes.
    xmp.spawn(worker_fn, nprocs=num_cores, args=(config,))


if __name__ == "__main__":
    main()
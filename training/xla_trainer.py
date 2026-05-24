#!/usr/bin/env python3
"""
XLA-aware trainer for SabiYarn.

Subclasses SabiYarnTrainer and overrides only the CUDA-specific methods.
All model architecture, loss computation, LR scheduling, evaluation logic,
and W&B logging are inherited unchanged.

Not a standalone script — imported by tpu_launch.py.

Method override map
-------------------
setup_environment          XLA device, nullcontext, no-op scaler, disable CCE
setup_logging              Set master_process from dist rank before wandb init
setup_distributed          Read from live process group, wrap model in DDP
setup_compilation          No-op (XLA is the compiler)
setup_distributed_dataloader  No pin_memory, wrap with MpDeviceLoader
get_batch                  Always use distributed loader path; skip debug .item() calls
monitor_and_control_gradients  No-op (per-step .item() calls would sync XLA graph each step)
safe_optimizer_step        xm.optimizer_step instead of scaler.step
estimate_loss              Batch graph execution across eval iterations before reading values
save_checkpoint_wandb      xm.save for XLA-aware tensor transfer + sync
"""

import os
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import structlog

from training.new_train import SabiYarnTrainer, TrainingConfig
from training.constant_tokens import MASK

LOG = structlog.stdlib.get_logger()


# ---------------------------------------------------------------------------
# No-op GradScaler
# ---------------------------------------------------------------------------

class _NoOpScaler:
    """
    Drop-in replacement for GradScaler on XLA.

    TPU natively supports bfloat16, which has the same exponent range as
    float32 and doesn't underflow the way float16 does on CUDA. Loss scaling
    is neither needed nor valid on XLA.

    Providing this object lets the inherited train() loop call
    scaler.scale(loss).backward() and scaler.update() without modification.
    safe_optimizer_step() is overridden separately to use xm.optimizer_step.
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


# ---------------------------------------------------------------------------
# XLA trainer
# ---------------------------------------------------------------------------

class XLASabiYarnTrainer(SabiYarnTrainer):
    """
    SabiYarnTrainer adapted for XLA / TPU.

    Prerequisite: torch.distributed must already be initialised with
    backend='xla' before this class is constructed. tpu_launch.py's
    worker_fn handles this.
    """

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def setup_environment(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = xm.xla_device()
        self.device_type = "xla"
        # config.device is a str used by inherited methods; keep it in sync.
        self.config.device = str(self.device)   # e.g. "xla:0"

        # bfloat16 is native on TPU — no autocast wrapper or loss scaling needed.
        self.ptdtype = torch.bfloat16
        self.ctx = nullcontext()

        # cut_cross_entropy uses Triton CUDA kernels; not available on XLA.
        self.config.use_cut_cross_entropy = False

        # The inherited train() loop calls self.scaler.scale(loss).backward().
        # A no-op scaler lets that call through as plain loss.backward().
        self.scaler = _NoOpScaler()

    # ------------------------------------------------------------------
    # Logging — must run before wandb.init
    # ------------------------------------------------------------------

    def setup_logging(self) -> None:
        # dist is already initialised by the time this runs on XLA, so we
        # can correctly gate wandb initialisation on rank 0 here — unlike the
        # GPU path where setup_distributed() hasn't run yet at this point.
        self.master_process = (dist.get_rank() == 0)
        super().setup_logging()

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------

    def setup_distributed(self) -> None:
        """
        torch.distributed is already initialised with backend='xla' by the
        launcher. We only need to read rank/world_size and wrap the model.
        """
        if not self.config.dist:
            self.master_process = True
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.master_process = (rank == 0)
        self.seed_offset = rank
        torch.manual_seed(1337 + self.seed_offset)

        # DDP calls dist.all_reduce() for gradient synchronisation. With
        # backend='xla', those calls route through XLA's collective ops.
        self.model = DDP(self.model)

        self.tokens_per_iter = (
            self.config.gradient_accumulation_steps
            * world_size
            * self.config.train_batch_size
            * self.config.max_seq_len
        )

        if self.master_process:
            LOG.info(
                "XLA DDP initialised",
                rank=rank,
                world_size=world_size,
                tokens_per_iter=self.tokens_per_iter,
            )

    # ------------------------------------------------------------------
    # Compilation — no-op on XLA
    # ------------------------------------------------------------------

    def setup_compilation(self) -> None:
        # XLA is the compiler. torch.compile targets CUDA/CPU backends and
        # conflicts with XLA's own graph tracing — do not apply it.
        pass

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def setup_distributed_dataloader(self) -> None:
        """
        Build DataLoaders without pin_memory (a CUDA-only concept), then
        wrap them with pl.MpDeviceLoader.

        MpDeviceLoader prefetches each batch onto the XLA device in a
        background thread, hiding the host→device transfer latency behind
        the previous step's compute. Iterating the wrapped loader yields
        tensors already on self.device — no explicit .to(device) in get_batch.

        drop_last=True on the training loader is mandatory: XLA recompiles
        the execution graph whenever a tensor shape changes. A partial final
        batch with a different batch size would trigger a full recompile.
        """

        class _TokenDataset(Dataset):
            def __init__(self, data_path: str, max_seq_len: int):
                self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
                self.max_seq_len = max_seq_len

            def __len__(self) -> int:
                # last window would be incomplete
                return len(self.data) - self.max_seq_len

            def __getitem__(self, idx: int):
                x = torch.from_numpy(
                    self.data[idx : idx + self.max_seq_len].astype(np.int64)
                )
                y = torch.from_numpy(
                    self.data[idx + 1 : idx + 1 + self.max_seq_len].astype(np.int64)
                )
                return x, y

        def _make_loader(path: str, shuffle: bool, drop_last: bool) -> DataLoader:
            dataset = _TokenDataset(path, self.config.max_seq_len)
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed_offset,
                drop_last=drop_last,
            )
            return DataLoader(
                dataset,
                batch_size=self.config.train_batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=False,
                drop_last=drop_last,
            )

        raw_train = _make_loader(
            self.config.train_data_path, shuffle=True, drop_last=True
        )
        raw_val = _make_loader(
            self.config.eval_data_path, shuffle=False, drop_last=False
        )

        self.train_loader = pl.MpDeviceLoader(raw_train, self.device)
        self.val_loader = pl.MpDeviceLoader(raw_val, self.device)
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        if self.master_process:
            LOG.info(
                "XLA data loaders ready",
                train_samples=len(raw_train.dataset),
                val_samples=len(raw_val.dataset),
                batch_size=self.config.train_batch_size,
                world_size=dist.get_world_size(),
            )

    def get_batch(self, split: str):
        """
        Return the next (x, y) pair for the given split.

        Tensors arrive pre-loaded on self.device via MpDeviceLoader —
        no .to(device) call needed. The base class debug checks
        (.item() calls on every batch) are intentionally omitted here:
        each .item() is a full XLA graph sync and would dominate step time
        if called unconditionally every batch.
        """
        loader = self.train_loader if split == "train" else self.val_loader
        iterator = "train_iter" if split == "train" else "val_iter"

        try:
            x, y = next(getattr(self, iterator))
        except StopIteration:
            setattr(self, iterator, iter(loader))
            x, y = next(getattr(self, iterator))

        return x, y

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def safe_optimizer_step(self) -> None:
        """
        XLA optimizer step.

        xm.optimizer_step does three things:
          1. Calls optimizer.step() inside the XLA execution context.
          2. Calls xm.mark_step() — compiles and executes the accumulated
             HLO graph, then clears it for the next iteration.

        Gradient clipping triggers one graph sync to compute the global norm.
        This is acceptable since the sync is immediately followed by the step.
        With DDP, all-reduce of gradients has already happened during backward
        via DDP's hooks; xm.optimizer_step doesn't re-reduce.
        """
        if self.config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        xm.optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Gradient monitoring — disabled on XLA
    # ------------------------------------------------------------------

    def monitor_and_control_gradients(
        self, model, optimizer, step: int, old_lr: float, **kwargs
    ) -> float:
        """
        The base class inspects .min()/.max()/.mean() on every parameter's
        gradient tensor, which requires .item() — a full XLA graph sync per
        call. Doing this every training step would dominate runtime.

        On XLA we skip per-step gradient inspection entirely and rely on
        grad_clip in safe_optimizer_step to keep gradients bounded. W&B
        gradient logging (log_grad_norm) still runs at log_interval via
        log_advanced_metrics, which is infrequent enough to be acceptable.
        """
        return old_lr

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Evaluate on train and val splits.

        Rather than calling .item() inside the loop (one graph sync per
        iteration = eval_iters compilations), we accumulate XLA tensors and
        flush the graph once per split. This way the entire eval split
        executes as a single compiled graph.
        """
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = []
            for _ in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    loss, _ = self.compute_loss(X, Y)
                losses.append(loss.detach())

            # Single mark_step flushes all eval iterations at once,
            # then one .item() reads the final scalar.
            xm.mark_step()
            out[split] = torch.stack(losses).mean().item()

        self.model.train()
        return out

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint_wandb(self) -> None:
        """
        Save checkpoint using xm.save().

        xm.save() handles:
          - An implicit mark_step() to flush any pending XLA ops.
          - Transferring tensors from XLA device memory to CPU.
          - Writing to disk on the master process only (master_only=True).

        master_only=True prevents the thundering-herd problem on pod slices
        where all hosts share a network filesystem (e.g. GCS FUSE).
        """
        if not self.master_process:
            return

        raw_model = self.model.module if self.config.dist else self.model
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": raw_model.params,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }

        os.makedirs(self.run_dir, exist_ok=True)
        ckpt_path = os.path.join(self.run_dir, "ckpt.pt")

        xm.save(checkpoint, ckpt_path, master_only=True)
        LOG.info("Checkpoint saved", path=ckpt_path, step=self.iter_num)

        try:
            with open(os.path.join(self.config.out_dir, "LATEST_RUN.txt"), "w") as fp:
                fp.write(self.run_dir)
        except Exception:
            pass

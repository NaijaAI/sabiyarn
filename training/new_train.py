#!/usr/bin/env python3
"""
Training script for SabiYarn models with support for:
- All attention mechanisms (MHA, MLA, Differential Attention)
- MoE, Multi-Token Prediction, Layer Sharing
- Custom causal masking
- Auto-distributed training detection
"""

import os
import time
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import structlog

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import SGD, Adam, AdamW
import numpy as np
import wandb

# SabiYarn imports
from ..data import prepare
from ..sabiyarn.model import ModelArgs, SabiYarn, AttentionType
from ..sabiyarn.MLA import MLAConfig
from ..sabiyarn.differential_attention import DiffAttnArgs
from ..cut_cross_entropy import linear_cross_entropy
from .utils import *
from .constant_tokens import MASK
from .training_attention_mask import create_causal_mask

try:
    from transformers import AutoTokenizer
except ImportError:
    print("⚠️ transformers not available, tokenizer features disabled")
    AutoTokenizer = None

try:
    from bitsandbytes import optim as bnb_optim
except ImportError:
    print("⚠️ bitsandbytes not available, 8-bit optimization disabled")
    bnb_optim = None

LOG = structlog.stdlib.get_logger()

@dataclass
class TrainingConfig:
    """Comprehensive training configuration for SabiYarn models."""
    
    # Model Architecture
    attention_type: str = "MLA"  # "self_attention", "differential_attention", "MLA"
    dim: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 52050
    max_seq_len: int = 1024
    max_batch_size: int = 32
    
    # Attention-specific configs
    use_mla: bool = True
    use_differential_attention: bool = False
    
    # MLA Configuration
    mla_q_lora_rank: int = 512
    mla_kv_lora_rank: int = 256
    mla_qk_rope_head_dim: int = 64
    mla_v_head_dim: int = 128
    mla_qk_nope_head_dim: int = 128
    
    # MoE Configuration (only with MLA)
    use_moe: bool = True
    n_routed_experts: int = 16
    n_activated_experts: int = 8
    moe_inter_dim: int = 2048
    n_shared_experts: int = 1
    score_function: str = "sigmoid"
    bias_update_speed: float = 0.001
    moe_aux_loss_weight: float = 0.001  # Weight for MoE sequence-wise auxiliary loss
    
    # Multi-Token Prediction (only with MLA)
    use_multi_token_prediction: bool = True
    num_prediction_tokens: int = 2
    mtp_loss_weight: float = 1.0  
    mtp_share_embeddings: bool = True
    mtp_only_training: bool = True  # When True, only use MTP loss for training
    
    # Layer Sharing (MobileLLM-style)
    use_layer_sharing: bool = True
    n_unique_layers: Optional[int] = 10
    
    # Other model features
    use_logic_network: bool = False
    use_j_linear: bool = True
    tie_weights: bool = True
    norm_eps: float = 1e-5
    init_std: float = 0.006
    
    # Training Configuration
    train_batch_size: int = 24
    gradient_accumulation_steps: int = 40  # 5 * 8
    learning_rate: float = 3e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 1500
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    
    # Optimizer
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd", "adam8bit"
    
    # Loss function
    use_cut_cross_entropy: bool = True
    
    # Custom masking
    use_custom_causal_mask: bool = True
    mask_id_value: int = 30  # ID value for custom masking
    
    # Data
    dataset: str = "Aletheia-ng/pretrain_test"
    train_data_path: str = "./train.bin"
    eval_data_path: str = "./val.bin"
    
    # Logging and checkpointing
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 100
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # "scratch" or "resume"
    
    # WandB logging
    wandb_log: bool = True
    wandb_project: str = "sabiyarn-new-training"
    wandb_run_name: str = "modern_training"
    
    # Generation testing
    display_model_output_iter: int = 768
    generation_max_tokens: int = 100
    
    # System
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float32", "bfloat16", "float16"
    compile_model: bool = True
    
    # Distributed training (auto-detected by model)
    auto_detect_distributed: bool = True


class SabiYarnTrainer:
    """Modern trainer for SabiYarn models with comprehensive feature support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.setup_distributed()
        self.setup_logging()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_compilation()
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.local_iter_num = 0
        self.running_mfu = -1.0
        
    def setup_environment(self):
        """Setup environment variables and device configuration."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set up device and dtype
        self.device_type = "cuda" if "cuda" in self.config.device else "cpu"
        
        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Configure precision
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.dtype]
        
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )
        
        # Initialize gradient scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))
        
    def setup_distributed(self):
        """Setup distributed training if available."""
        # Let the model auto-detect distributed configuration
        self.ddp = False
        self.master_process = True
        self.seed_offset = 0
        self.ddp_world_size = 1
        
        # Check if we should initialize DDP manually
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                backend = "nccl"
                rank = int(os.environ.get("RANK", 0))
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                
                init_process_group(backend=backend)
                torch.cuda.set_device(local_rank)
                
                self.ddp = True
                self.master_process = rank == 0
                self.seed_offset = rank
                self.ddp_world_size = world_size
                self.local_rank = local_rank
                
                # Adjust gradient accumulation for distributed training
                assert self.config.gradient_accumulation_steps % world_size == 0
                self.config.gradient_accumulation_steps //= world_size
                
                LOG.info(f"Distributed training initialized: rank {rank}/{world_size}")
        
        # Calculate tokens per iteration
        self.tokens_per_iter = (
            self.config.gradient_accumulation_steps * 
            self.ddp_world_size * 
            self.config.train_batch_size * 
            self.config.max_seq_len
        )
        LOG.info(f"Tokens per iteration: {self.tokens_per_iter:,}")
        
        # Set random seed
        torch.manual_seed(1337 + self.seed_offset)
        
    def setup_logging(self):
        """Setup output directory and wandb logging."""
        if self.master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)
            
            if self.config.wandb_log:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__
                )
                LOG.info("WandB logging initialized")
                
    def setup_data(self):
        """Setup data loading."""
        LOG.info("Preparing dataset...")
        prepare.run()
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("Aletheia-ng/SabiYarn-125M")
                LOG.info("Tokenizer loaded successfully")
            except Exception as e:
                LOG.warning(f"Could not load tokenizer: {e}")
                
    def setup_model(self):
        """Setup the SabiYarn model with all configurations."""
        LOG.info("Initializing model...")
        
        if self.config.init_from == "scratch":
            # Create model configuration based on attention type
            model_args = self.create_model_args()
            self.model = SabiYarn(model_args)
            LOG.info(f"Model initialized from scratch: {self.model.get_model_size()}")
            
        elif self.config.init_from == "resume":
            # Load from checkpoint
            ckpt_path = os.path.join(self.config.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.config.device)
            
            model_args = checkpoint["model_args"]
            self.model = SabiYarn(model_args)
            self.model.load_state_dict(checkpoint["model"])
            
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            
            LOG.info(f"Model resumed from checkpoint: {self.model.get_model_size()}")
            
        self.model.to(self.config.device)
        
    def create_model_args(self) -> ModelArgs:
        """Create ModelArgs based on training configuration."""
        
        # Create attention-specific configurations
        mla_config = None
        diff_attn_args = None
        
        if self.config.attention_type == "MLA":
            mla_config = MLAConfig(
                hidden_size=self.config.dim,
                num_heads=self.config.n_heads,
                max_seq_len=self.config.max_seq_len,
                max_batch_size=self.config.max_batch_size,
                attention_dropout=0.0,
                q_lora_rank=self.config.mla_q_lora_rank,
                qk_rope_head_dim=self.config.mla_qk_rope_head_dim,
                kv_lora_rank=self.config.mla_kv_lora_rank,
                v_head_dim=self.config.mla_v_head_dim,
                qk_nope_head_dim=self.config.mla_qk_nope_head_dim,
                attention_bias=False,
                original_seq_len=self.config.max_seq_len,
                rope_theta=10000.0,
                rope_factor=1,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0
            )
            
        elif self.config.attention_type == "differential_attention":
            diff_attn_args = DiffAttnArgs(
                depth=0,  # Will be set per layer
                max_batch_size=self.config.max_batch_size,
                n_heads=self.config.n_heads,
                embed_dim=self.config.dim,
                n_kv_heads=self.config.n_kv_heads or self.config.n_heads,
                max_seq_len=self.config.max_seq_len,
                norm_eps=self.config.norm_eps
            )
        
        return ModelArgs(
            # Basic architecture
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            vocab_size=self.config.vocab_size,
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            
            # Attention configuration
            attention_type=getattr(AttentionType, self.config.attention_type.upper()),
            mla_config=mla_config,
            diff_attn_args=diff_attn_args,
            
            # MoE configuration (only with MLA)
            moe=self.config.use_moe and self.config.attention_type == "MLA",
            n_routed_experts=self.config.n_routed_experts,
            n_activated_experts=self.config.n_activated_experts,
            moe_inter_dim=self.config.moe_inter_dim,
            n_shared_experts=self.config.n_shared_experts,
            bias_update_speed=self.config.bias_update_speed,
            moe_aux_loss_weight=self.config.moe_aux_loss_weight,
            score_function=self.config.score_function,
            
            # Multi-token prediction (only with MLA)
            multi_token_prediction=self.config.use_multi_token_prediction and self.config.attention_type == "MLA",
            num_prediction_tokens=self.config.num_prediction_tokens,
            mtp_loss_weight=self.config.mtp_loss_weight,
            mtp_share_embeddings=self.config.mtp_share_embeddings,
            
            # Layer sharing
            layer_sharing=self.config.use_layer_sharing,
            n_unique_layers=self.config.n_unique_layers,
            
            # Other features
            logic_network=self.config.use_logic_network,
            use_j=self.config.use_j_linear,
            tie_weights=self.config.tie_weights,
            norm_eps=self.config.norm_eps,
            init_std=self.config.init_std,
            
            # Distributed training (auto-detected)
            auto_detect_distributed=self.config.auto_detect_distributed,
        )
        
    def setup_optimizer(self):
        """Setup optimizer based on configuration."""
        if self.config.optimizer_type == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adam8bit" and bnb_optim is not None:
            self.optimizer = bnb_optim.Adam8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
            
        # Load optimizer state if resuming
        if self.config.init_from == "resume":
            ckpt_path = os.path.join(self.config.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.config.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        LOG.info(f"Optimizer initialized: {self.config.optimizer_type}")
        
    def setup_compilation(self):
        """Setup model compilation and DDP wrapping."""
        # Compile model if requested
        if self.config.compile_model:
            LOG.info("Compiling model...")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)
            
        # Wrap in DDP if needed
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
    def get_batch(self, split: str):
        """Load a batch of data."""
        data_path = self.config.train_data_path if split == "train" else self.config.eval_data_path
        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        
        ix = torch.randint(len(data) - self.config.max_seq_len, (self.config.train_batch_size,))
        x = [torch.from_numpy((data[i:i + self.config.max_seq_len]).astype(np.int64)) for i in ix]
        y = [torch.from_numpy((data[i + 1:i + 1 + self.config.max_seq_len]).astype(np.int64)) for i in ix]
        
        # Apply label processing
        y = [mask_long_sequences(process_labels(sample.clone(), MASK), mask_value=MASK) for sample in y]
        
        x = torch.stack(x)
        y = torch.stack(y)
        
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.config.device, non_blocking=True)
            y = y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            
        return x, y
        
    def prepare_attention_mask(self, tokens: torch.Tensor):
        """Prepare attention mask with optional custom causal masking."""
        batch_size, seq_len = tokens.shape
        
        # Create standard causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))
        mask = mask.view(1, 1, seq_len, seq_len).repeat(batch_size, 1, 1, 1)
        
        # Apply custom causal masking if enabled
        if self.config.use_custom_causal_mask:
            mask = create_causal_mask(
                tokens, 
                mask, 
                id_val=self.config.mask_id_value
            )
            
        return mask
        
    def compute_loss(self, tokens: torch.Tensor, targets: torch.Tensor):
        """Compute loss with support for different loss functions and multi-token prediction."""
        # Prepare attention mask
        mask = self.prepare_attention_mask(tokens)
        
        # Forward pass
        if self.model.use_multi_token:
            hidden_states, logits, multi_token_logits = self.model(
                tokens, start_pos=0, mask=mask, return_multi_token=True
            )
            
            # When MTP is enabled, use ONLY MTP loss to train both main model and MTP modules
            from ..sabiyarn.multi_token_loss import MultiTokenLoss
            
            mtp_loss_fn = MultiTokenLoss(
                num_prediction_tokens=self.config.num_prediction_tokens,
                mtp_loss_weight=self.config.mtp_loss_weight,
                use_cut_cross_entropy=self.config.use_cut_cross_entropy
            )
            
            # Create extended targets for MTP
            extended_targets = F.pad(targets, (0, self.config.num_prediction_tokens), value=-100)
            
            try:
                # Get MTP hidden states and output heads for Cut Cross Entropy
                raw_model = self.model.module if self.ddp else self.model
                mtp_hidden_states = None
                mtp_output_heads = None
                
                if hasattr(raw_model, 'multi_token_predictor') and raw_model.multi_token_predictor is not None:
                    # Get the final hidden states from MTP module
                    mtp_module = raw_model.multi_token_predictor
                    if hasattr(mtp_module, 'output_norm'):
                        # This would be the normalized output from MTP transformer block
                        # We'll need to modify the forward pass to return this
                        pass
                    mtp_output_heads = mtp_module.output_heads
                
                total_loss = mtp_loss_fn(
                    multi_token_logits, 
                    extended_targets,
                    mtp_hidden_states=mtp_hidden_states,
                    mtp_output_heads=mtp_output_heads
                )
            except Exception as e:
                LOG.warning(f"MTP loss computation failed: {e}")
                # Fallback to standard loss
                if self.config.use_cut_cross_entropy:
                    total_loss = linear_cross_entropy(
                        hidden_states,
                        raw_model.lm_head.weight,
                        targets,
                        shift=True,
                        impl="torch_compile"
                    )
                else:
                    total_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100
                    )
        else:
            # Standard training without MTP
            hidden_states, logits = self.model(tokens, start_pos=0, mask=mask)
            
            if self.config.use_cut_cross_entropy:
                raw_model = self.model.module if self.ddp else self.model
                total_loss = linear_cross_entropy(
                    hidden_states,
                    raw_model.lm_head.weight,
                    targets,
                    shift=True,
                    impl="torch_compile"
                )
            else:
                total_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                
        return total_loss, logits if not self.model.use_multi_token else hidden_states
        
    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and validation sets."""
        out = {}
        self.model.eval()
        
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                
                with self.ctx:
                    loss, _ = self.compute_loss(X, Y)
                    losses[k] = loss.item()
                    
            out[split] = losses.mean()
            
        self.model.train()
        return out
        
    def get_lr(self, it: int) -> float:
        """Learning rate schedule with warmup and cosine decay."""
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
            
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
        
    def generate_sample_text(self, tokens: torch.Tensor):
        """Generate sample text for monitoring training progress."""
        if self.tokenizer is None:
            return
            
        self.model.eval()
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    tokens[:1],  # Use first sample
                    max_new_tokens=self.config.generation_max_tokens,
                    use_multi_token=self.model.use_multi_token
                )
                
            input_text = self.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
            output_text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            
            LOG.info("=" * 50)
            LOG.info(f"Input: {input_text[-100:]}")
            LOG.info(f"Generated: {output_text[len(input_text):]}")
            LOG.info("=" * 50)
            
        except Exception as e:
            LOG.warning(f"Text generation failed: {e}")
        finally:
            self.model.train()
            
    def save_checkpoint(self):
        """Save training checkpoint."""
        if not self.master_process:
            return
            
        raw_model = self.model.module if self.ddp else self.model
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": raw_model.params,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        
        # Save main checkpoint
        ckpt_path = os.path.join(self.config.out_dir, "ckpt.pt")
        torch.save(checkpoint, ckpt_path)
        LOG.info(f"Checkpoint saved to {ckpt_path}")
        
        # Save MTP modules separately if available
        if hasattr(raw_model, 'multi_token_predictor') and raw_model.multi_token_predictor is not None:
            mtp_checkpoint = {
                "mtp_state_dict": raw_model.multi_token_predictor.state_dict(),
                "model_args": raw_model.params,
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.config.__dict__,
            }
            
            mtp_ckpt_path = os.path.join(self.config.out_dir, "mtp_ckpt.pt")
            torch.save(mtp_checkpoint, mtp_ckpt_path)
            LOG.info(f"MTP module checkpoint saved to {mtp_ckpt_path}")
            
            # Also save individual MTP components for fine-grained control
            mtp_components = {}
            mtp_module = raw_model.multi_token_predictor
            
            if hasattr(mtp_module, 'mtp_transformer_block'):
                mtp_components['transformer_block'] = mtp_module.mtp_transformer_block.state_dict()
                
            if hasattr(mtp_module, 'output_heads') and mtp_module.output_heads is not None:
                mtp_components['output_heads'] = mtp_module.output_heads.state_dict()
                
            if hasattr(mtp_module, 'projection'):
                mtp_components['projection'] = mtp_module.projection.state_dict()
                
            if mtp_components:
                mtp_components_path = os.path.join(self.config.out_dir, "mtp_components.pt")
                torch.save(mtp_components, mtp_components_path)
                LOG.info(f"MTP components saved to {mtp_components_path}")
        
    def train(self):
        """Main training loop."""
        LOG.info("Starting training...")
        
        X, Y = self.get_batch("train")
        t0 = time.time()
        
        while True:
            # Update learning rate
            lr = self.get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
                
            # Evaluation and checkpointing
            if self.iter_num % self.config.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                LOG.info(f"Step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": self.running_mfu * 100,
                    })
                    
                if losses["val"] < self.best_val_loss or self.config.always_save_checkpoint:
                    self.best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        self.save_checkpoint()
                        
            if self.iter_num == 0 and self.config.eval_only:
                break
                
            # Generate sample text occasionally
            if (self.iter_num % self.config.display_model_output_iter == 0 and 
                self.master_process and self.iter_num > 0):
                self.generate_sample_text(X)
                
            # Training step with gradient accumulation
            for micro_step in range(self.config.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.config.gradient_accumulation_steps - 1
                    )
                    
                with self.ctx:
                    loss, _ = self.compute_loss(X, Y)
                    loss = loss / self.config.gradient_accumulation_steps
                    
                # Get next batch while GPU is busy
                X, Y = self.get_batch("train")
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
            # Gradient clipping and optimizer step
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            if self.iter_num % self.config.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.config.gradient_accumulation_steps
                LOG.info(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                
            self.iter_num += 1
            self.local_iter_num += 1
            
            # Termination condition
            if self.iter_num > self.config.max_iters:
                break
                
        if self.ddp:
            destroy_process_group()
            
        LOG.info("Training completed!")


def main():
    """Main entry point."""
    # Create training configuration
    config = TrainingConfig()
    
    # Override config from environment or command line if needed
    # You can add argparse here or load from config files
    
    # Initialize and run trainer
    trainer = SabiYarnTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
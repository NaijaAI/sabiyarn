#!/usr/bin/env python3
"""
Training script for SabiYarn models with support for:
- All attention mechanisms (MHA, MLA, Differential Attention)
- MoE, Multi-Token Prediction, Layer Sharing
- Custom causal masking
- Auto-distributed training detection
"""

import os
import sys
import time
import math
from datetime import datetime
import json
import random
import string
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_FILE_LOADED = True
except ImportError:
    ENV_FILE_LOADED = False

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
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
from data import prepare
from sabiyarn.model import ModelArgs, SabiYarn, AttentionType
from sabiyarn.MLA import MLAConfig
from sabiyarn.differential_attention import DiffAttnArgs
from cut_cross_entropy import linear_cross_entropy
from training.utils import *
from training.constant_tokens import MASK
from training.training_attention_mask import create_causal_mask, create_causal_mask_optimized

from transformers import AutoTokenizer
from bitsandbytes import optim as bnb_optim


try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    print("⚠️ psutil/GPUtil not available, system monitoring disabled")
    MONITORING_AVAILABLE = False

LOG = structlog.stdlib.get_logger()

@dataclass
class TrainingConfig:
    # Model Architecture
    attention_type: str = "MLA"  # "self_attention", "differential_attention", "MLA"
    dim: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 64000
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
    layer_sharing_strategy: str = "immediate"
    layer_sharing: bool = True
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
    mask_id_value: int = 1  # ID value for custom masking, this is for the end of text token id value from our tokenizer which is 1 for llama 
    
    # Data
    dataset: str = "Aletheia-ng/pretrain_test"
    train_data_path: str = "./train.bin"
    eval_data_path: str = "./val.bin"
    
    # Logging and checkpointing
    out_dir: str = "out"  # Base directory that will contain per-run subfolders
    run_dir: Optional[str] = None  # Full path to the current run directory (auto-created if None)
    resume_run_dir: Optional[str] = None  # When resuming, explicitly set the run folder
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
    wandb_tags: list = field(default_factory=lambda: ["MLA", "MoE", "MTP", "SabiYarn"])
    
    # Advanced monitoring
    log_grad_norm: bool = True
    log_weights: bool = True  # Log weight distributions
    log_system_metrics: bool = True
    log_moe_metrics: bool = True  # Log MoE expert utilization
    log_attention_metrics: bool = True  # Log attention statistics
    monitor_interval: int = 50  # Log detailed metrics every N steps
    
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
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.setup_distributed()
        self.setup_output_dirs()
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
        self.step_start_time = time.time()
        
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
        self.scaler = torch.amp.GradScaler('cuda',enabled=(self.config.dtype == "float16"))
        
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
            # Ensure run directory exists
            os.makedirs(self.run_dir, exist_ok=True)
            
            if self.config.wandb_log:
                # Check W&B authentication
                self._check_wandb_auth()
                
                # Create dynamic run name with timestamp and key parameters
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{self.config.wandb_run_name}_{self.config.attention_type}"
                if self.config.use_moe:
                    run_name += f"_moe{self.config.n_routed_experts}x{self.config.n_activated_experts}"
                if self.config.use_multi_token_prediction:
                    run_name += f"_mtp{self.config.num_prediction_tokens}"
                run_name += f"_{timestamp}"
                
                # Enhanced W&B config
                wandb_config = self.config.__dict__.copy()
                wandb_config.update({
                    "model_size": "TBD",  # Will be updated after model creation
                    "dataset_name": self.config.dataset,
                    "hardware": self._get_hardware_info(),
                    "git_commit": self._get_git_commit(),
                    "run_dir": self.run_dir,
                })
                
                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    config=wandb_config,
                    tags=self.config.wandb_tags,
                    notes=f"SabiYarn training with {self.config.attention_type} attention",
                    save_code=True
                )
                LOG.info(f"WandB logging initialized: {run_name}")
    
    def _check_wandb_auth(self):
        """Check W&B authentication and provide helpful guidance."""
        try:
            # Try to get API key from various sources
            api_key = (
                os.getenv("WANDB_API_KEY") or 
                wandb.api.api_key or
                None
            )
            
            if not api_key:
                LOG.warning("⚠️ W&B API key not found!")
                LOG.info("🔧 To authenticate with W&B, choose one of these methods:")
                LOG.info("   1. Run: wandb login")
                LOG.info("   2. Set environment variable: export WANDB_API_KEY=your_key")
                LOG.info("   3. Create .env file with: WANDB_API_KEY=your_key")
                LOG.info("   4. Get your key from: https://wandb.ai/authorize")
                
                # Try to authenticate interactively if possible
                try:
                    wandb.login()
                    LOG.info("✅ W&B authentication successful!")
                except Exception as e:
                    LOG.error(f"❌ W&B authentication failed: {e}")
                    LOG.info("💡 Continuing without W&B logging...")
                    self.config.wandb_log = False
                    return
            else:
                LOG.info("✅ W&B API key found")
                
        except Exception as e:
            LOG.warning(f"⚠️ W&B authentication check failed: {e}")
            LOG.info("💡 Continuing with W&B - it may prompt for login...")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for logging."""
        info = {}
        
        if MONITORING_AVAILABLE:
            try:
                info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                })
            except:
                pass
        
        if torch.cuda.is_available():
            try:
                info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    "cuda_version": torch.version.cuda,
                })
            except:
                pass
        
        return info
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(__file__)
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {}
        
        if not MONITORING_AVAILABLE:
            return metrics
        
        try:
            # CPU and Memory
            metrics["system/cpu_percent"] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            metrics["system/memory_percent"] = memory.percent
            metrics["system/memory_available_gb"] = memory.available / (1024**3)
        except:
            pass
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    metrics["system/gpu_utilization"] = gpu.load * 100
                    metrics["system/gpu_memory_percent"] = gpu.memoryUtil * 100
                    metrics["system/gpu_temperature"] = gpu.temperature
            except:
                pass  # GPU monitoring failed, continue without
                
            try:
                # PyTorch GPU memory
                metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            except:
                pass
        
        return metrics
    
    def get_moe_metrics(self, model) -> Dict[str, float]:
        """Get MoE-specific metrics."""
        metrics = {}
        
        if not self.config.use_moe:
            return metrics
            
        raw_model = model.module if self.ddp else model
        
        # Find MoE layers
        moe_layers = []
        for name, module in raw_model.named_modules():
            if hasattr(module, 'gate') and hasattr(module.gate, 'expert_bias'):
                moe_layers.append((name, module))
        
        if moe_layers:
            # Expert bias statistics
            all_biases = []
            for name, moe_layer in moe_layers:
                expert_bias = moe_layer.gate.expert_bias.data
                all_biases.append(expert_bias)
                
                # Per-layer metrics
                metrics[f"moe/{name}/expert_bias_mean"] = expert_bias.mean().item()
                metrics[f"moe/{name}/expert_bias_std"] = expert_bias.std().item()
                metrics[f"moe/{name}/expert_bias_max"] = expert_bias.max().item()
                metrics[f"moe/{name}/expert_bias_min"] = expert_bias.min().item()
            
            # Global MoE metrics
            if all_biases:
                global_bias = torch.cat(all_biases)
                metrics["moe/global_expert_bias_mean"] = global_bias.mean().item()
                metrics["moe/global_expert_bias_std"] = global_bias.std().item()
        
        return metrics
    
    def get_gradient_metrics(self, model) -> Dict[str, float]:
        """Get gradient statistics."""
        metrics = {}
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log gradients for key components
                if any(key in name for key in ['gate', 'expert', 'attention', 'lm_head']):
                    metrics[f"grad/{name.replace('.', '/')}_norm"] = param_norm.item()
        
        if param_count > 0:
            metrics["grad/global_norm"] = total_norm ** 0.5
            metrics["grad/param_count"] = param_count
        
        return metrics
    
    def get_weight_metrics(self, model) -> Dict[str, float]:
        """Get weight distribution statistics."""
        metrics = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                
                # Log statistics for key components
                if any(key in name for key in ['gate', 'expert', 'attention', 'lm_head', 'tok_embeddings']):
                    clean_name = name.replace('.', '/')
                    metrics[f"weights/{clean_name}_mean"] = weight_data.mean().item()
                    metrics[f"weights/{clean_name}_std"] = weight_data.std().item()
                    metrics[f"weights/{clean_name}_max"] = weight_data.max().item()
                    metrics[f"weights/{clean_name}_min"] = weight_data.min().item()
        
        return metrics
    
    def log_advanced_metrics(self, loss: torch.Tensor, model, optimizer):
        """Log comprehensive metrics for SOTA monitoring."""
        if not self.config.wandb_log or not self.master_process:
            return
        
        # Base metrics
        metrics = {
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/epoch": self.iter_num * self.config.train_batch_size / 50000,  # Approximate
            "train/step": self.iter_num,
        }
        
        # Performance metrics
        current_time = time.time()
        if hasattr(self, 'step_start_time'):
            step_time = current_time - self.step_start_time
            metrics["perf/tokens_per_sec"] = self.tokens_per_iter / step_time
            metrics["perf/step_time_ms"] = step_time * 1000
        self.step_start_time = current_time
        
        # Gradient monitoring
        if self.config.log_grad_norm:
            metrics.update(self.get_gradient_metrics(model))
        
        # Weight monitoring (less frequent)
        if self.config.log_weights and self.iter_num % (self.config.monitor_interval * 4) == 0:
            metrics.update(self.get_weight_metrics(model))
        
        # MoE-specific metrics
        if self.config.log_moe_metrics and self.config.use_moe:
            metrics.update(self.get_moe_metrics(model))
        
        # System metrics (less frequent)
        if self.config.log_system_metrics and self.iter_num % self.config.monitor_interval == 0:
            metrics.update(self.get_system_metrics())
        
        # Log to W&B
        wandb.log(metrics, step=self.iter_num)
                
    def setup_data(self):
        """Setup data loading."""
        LOG.info("Preparing dataset...")
        # Ensure prepare writes to the configured paths 
        try:
            os.environ["TRAIN_DATA_PATH"] = self.config.train_data_path
            os.environ["VAL_DATA_PATH"] = self.config.eval_data_path
        except Exception:
            pass
        prepare.run(["Aletheia-ng/pretrain_test"], os.cpu_count())
        
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
            model_size_info = self.model.get_model_size()
            LOG.info(f"Model initialized from scratch: {model_size_info}")
            
            # Update W&B with actual model size
            if self.config.wandb_log and self.master_process:
                wandb.config.update({"model_size": model_size_info}, allow_val_change=True)
            
        elif self.config.init_from == "resume":
            # Load from checkpoint
            # Prefer an explicit resume_run_dir, else use the trainer's run_dir
            resume_dir = self.config.resume_run_dir or self.run_dir
            ckpt_path = os.path.join(resume_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.config.device)
            
            model_args = checkpoint["model_args"]
            self.model = SabiYarn(model_args)
            self.model.load_state_dict(checkpoint["model"])
            
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            
            LOG.info(f"Model resumed from checkpoint: {self.model.get_model_size()}")
            
        self.model.to(device=torch.device(self.config.device), dtype=self.ptdtype)
        
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
            layer_sharing=self.config.layer_sharing,
            n_unique_layers=self.config.n_unique_layers,
            layer_sharing_strategy=self.config.layer_sharing_strategy,
            
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
        y = [process_labels_optimized(sample.clone(), MASK) for sample in y]
        
        x = torch.stack(x)
        y = torch.stack(y)
        
        # Debug: Check for invalid tokens in raw data
        if x.max() >= self.config.vocab_size:
            LOG.error(f"Invalid input tokens in batch! Max: {x.max()}, vocab_size: {self.config.vocab_size}")
            LOG.error(f"Invalid x values: {x[x >= self.config.vocab_size]}")
            
        if y.max() >= self.config.vocab_size:
            LOG.error(f"Invalid target tokens in batch! Max: {y.max()}, vocab_size: {self.config.vocab_size}")
            LOG.error(f"Invalid y values: {y[y >= self.config.vocab_size]}")
        
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
            mask = create_causal_mask_optimized(
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
            from sabiyarn.multi_token_loss import MultiTokenLoss
            
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
                        ignore_index=-100,
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
                    ignore_index=-100,
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
        
        # Save in the run directory
        os.makedirs(self.run_dir, exist_ok=True)
        # Update a rolling 'ckpt.pt' and also an iter-stamped file for history
        ckpt_latest = os.path.join(self.run_dir, "ckpt.pt")
        ckpt_iter = os.path.join(self.run_dir, f"ckpt_{self.iter_num:07d}.pt")
        torch.save(checkpoint, ckpt_latest)
        torch.save(checkpoint, ckpt_iter)
        # Update a simple pointer file in base out_dir for discovery
        try:
            with open(os.path.join(self.config.out_dir, "LATEST_RUN.txt"), "w") as fp:
                fp.write(self.run_dir)
        except Exception:
            pass
        LOG.info(f"Checkpoint saved to {ckpt_latest} and {ckpt_iter}")
        
        # Save MTP modules separately if available
        if hasattr(raw_model, 'multi_token_predictor') and raw_model.multi_token_predictor is not None:
            mtp_checkpoint = {
                "mtp_state_dict": raw_model.multi_token_predictor.state_dict(),
                "model_args": raw_model.params,
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.config.__dict__,
            }
            
            mtp_ckpt_path = os.path.join(self.run_dir, "mtp_ckpt.pt")
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
                mtp_components_path = os.path.join(self.run_dir, "mtp_components.pt")
                torch.save(mtp_components, mtp_components_path)
                LOG.info(f"MTP components saved to {mtp_components_path}")

    def setup_output_dirs(self):
        """Create and register a unique run directory under out_dir and write metadata/pointers.

        Naming: <YYYYmmdd_HHMMSS>_<attn>_<dim>d_<layers>L_<heads>H_<moe|dense>_<suffix>
        """
        # Ensure base exists
        os.makedirs(self.config.out_dir, exist_ok=True)

        if self.config.init_from == "resume":
            # If an explicit resume dir is given, use it; else attempt to detect latest
            if self.config.resume_run_dir:
                self.run_dir = self.config.resume_run_dir
            elif self.config.run_dir:
                self.run_dir = self.config.run_dir
            else:
                # Try pointer file first
                pointer = os.path.join(self.config.out_dir, "LATEST_RUN.txt")
                run_dir = None
                if os.path.exists(pointer):
                    try:
                        with open(pointer, "r") as fp:
                            candidate = fp.read().strip()
                            if candidate and os.path.isdir(candidate):
                                run_dir = candidate
                    except Exception:
                        run_dir = None
                if run_dir is None:
                    # Pick most recent directory under out_dir
                    subdirs = [d.path for d in os.scandir(self.config.out_dir) if d.is_dir()]
                    if not subdirs:
                        raise FileNotFoundError(f"No run directories found in {self.config.out_dir} to resume from")
                    run_dir = max(subdirs, key=lambda p: os.path.getmtime(p))
                self.run_dir = run_dir
        else:
            if self.config.run_dir:
                self.run_dir = self.config.run_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                attn = self.config.attention_type
                moe_tag = "moe" if self.config.use_moe else "dense"
                suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
                dir_name = f"{timestamp}_{attn}_{self.config.dim}d_{self.config.n_layers}L_{self.config.n_heads}H_{moe_tag}_{suffix}"
                self.run_dir = os.path.join(self.config.out_dir, dir_name)
            os.makedirs(self.run_dir, exist_ok=True)
            # Write metadata for this run
            metadata = {
                "created_at": datetime.now().isoformat(),
                "git_commit": self._get_git_commit(),
                "config": self.config.__dict__,
            }
            try:
                with open(os.path.join(self.run_dir, "metadata.json"), "w") as fp:
                    json.dump(metadata, fp, indent=2)
                with open(os.path.join(self.config.out_dir, "LATEST_RUN.txt"), "w") as fp:
                    fp.write(self.run_dir)
            except Exception:
                pass
        
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
                    eval_metrics = {
                        "iter": self.iter_num,
                        "eval/train_loss": losses["train"],
                        "eval/val_loss": losses["val"],
                        "train/lr": lr,
                        "perf/mfu_percent": self.running_mfu * 100,
                    }
                    
                    # Add system metrics during evaluation
                    if self.config.log_system_metrics:
                        eval_metrics.update(self.get_system_metrics())
                    
                    wandb.log(eval_metrics, step=self.iter_num)
                    
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
                
                # Log advanced metrics
                self.log_advanced_metrics(loss * self.config.gradient_accumulation_steps, self.model, self.optimizer)
                
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
    config = TrainingConfig()
    trainer = SabiYarnTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
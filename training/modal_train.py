 #!/usr/bin/env python3
"""
Modal GPU training wrapper for SabiYarn models.
"""

import modal

# Create Modal app with enhanced image for training
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "transformers",
        "wandb", 
        "structlog",
        "numpy",
        "psutil",
        "gputil",
        "python-dotenv",
        "bitsandbytes",
    ])
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app", ignore=[
        ".git", "*.pyc", "__pycache__", ".pytest_cache", "*.egg-info", 
        "out/", "*.bin", ".env"
    ])
)

app = modal.App("sabiyarn-training")

# Volume for persistent data and checkpoints
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)

@app.function(
    gpu="A100-40GB",  
    timeout=86400,  # 24 hours
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],  # Store W&B API key
    cpu=8,
    # memory=32768,  # 32GB RAM
)
def train_sabiyarn(
    # Model configuration
    attention_type: str = "self_attention",
    dim: int = 2048,
    n_layers: int = 20,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    vocab_size: int = 64000,
    max_seq_len: int = 1024,
    max_batch_size: int = 32,
    
    # MoE configuration
    use_moe: bool = False,
    n_routed_experts: int = 16,
    n_activated_experts: int = 8,
    moe_inter_dim: int = 2048,
    n_shared_experts: int = 1,
    score_function: str = "sigmoid",
    bias_update_speed: float = 0.001,
    moe_aux_loss_weight: float = 0.001,
    
    # Multi-Token Prediction
    use_multi_token_prediction: bool = True,
    num_prediction_tokens: int = 2,
    mtp_only_training: bool = True,
    
    # Layer Sharing
    layer_sharing: bool = True,
    layer_sharing_strategy: str = "immediate",
    n_unique_layers: int = 10,
    
    # Training configuration
    train_batch_size: int = 16,
    gradient_accumulation_steps: int = 5,
    learning_rate: float = 3e-4,
    max_iters: int = 60000,
    weight_decay: float = 1e-1,
    grad_clip: float = 1.0,
    warmup_iters: int = 300,
    lr_decay_iters: int = 1000,
    
    # Data and checkpointing
    dataset: str = "Aletheia-ng/pretrain_test",
    out_dir: str = "/data/checkpoints",
    eval_interval: int = 2000,
    log_interval: int = 100,

    init_from: str = "scratch",
    use_cut_cross_entropy=False,
    
    # W&B configuration
    wandb_project: str = "sabiyarn-modal-training",
    wandb_run_name: str = "modal_training",
    
    # System
    dtype: str = "bfloat16",
    compile_model: bool = True,
):
    """
    Train SabiYarn model on Modal GPU with comprehensive monitoring.
    
    Args:
        All the configuration parameters for the training run.
    """
    import os
    import sys
    
    # Setup paths for Modal environment
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    # Import training modules
    from training.new_train import TrainingConfig, SabiYarnTrainer
    
    # Create configuration
    config = TrainingConfig(
        # Model Architecture
        attention_type=attention_type,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        
        # MoE Configuration
        use_moe=use_moe,
        n_routed_experts=n_routed_experts,
        n_activated_experts=n_activated_experts,
        moe_inter_dim=moe_inter_dim,
        n_shared_experts=n_shared_experts,
        score_function=score_function,
        bias_update_speed=bias_update_speed,
        moe_aux_loss_weight=moe_aux_loss_weight,
        
        # Multi-Token Prediction
        use_multi_token_prediction=use_multi_token_prediction,
        num_prediction_tokens=num_prediction_tokens,
        mtp_only_training=mtp_only_training,
        
        # Layer Sharing
        layer_sharing=layer_sharing,
        layer_sharing_strategy=layer_sharing_strategy,
        n_unique_layers=n_unique_layers,
        
        #CCE
        use_cut_cross_entropy=use_cut_cross_entropy,
        
        # Training Configuration
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_iters=max_iters,
        weight_decay=weight_decay,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        grad_clip=grad_clip,
        
        # Data paths (Modal persistent volume)
        dataset=dataset,
        train_data_path="/data/train.bin",
        eval_data_path="/data/val.bin",
        out_dir=out_dir,
        eval_interval=eval_interval,
        log_interval=log_interval,
        run_dir=run_dir,
        
        # W&B Configuration
        wandb_log=True,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_tags=["Modal", "GPU", attention_type, "SabiYarn"],
        
        # Enhanced monitoring for Modal
        log_grad_norm=True,
        log_weights=True,
        log_system_metrics=True,
        log_moe_metrics=use_moe,
        monitor_interval=50,
        
        init_from=init_from,
        

        # System
        device="cuda",
        dtype=dtype,
        compile_model=True,  # Disable for debugging
    )
    
    print("Starting SabiYarn training on Modal GPU...")
    
    # Initialize and run trainer
    trainer = SabiYarnTrainer(config)
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("hf-secret")],
    cpu=8
)
def prepare_data():
    """Prepare training data on Modal."""
    import os
    import sys
    
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    from data import prepare
    
    print("📁 Preparing training data...")
    
    # Persist Hugging Face caches on the mounted Modal volume for stability
    cache_root = "/data/hf_cache"
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "transformers"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

    # Set data paths to persistent volume
    os.environ["TRAIN_DATA_PATH"] = "/data/train.bin"
    os.environ["VAL_DATA_PATH"] = "/data/val.bin"
    
    prepare.run(["Aletheia-ng/pretrain_test"],1)
    
    # Validate the created data files
    import numpy as np
    try:
        train_data = np.memmap("/data/train.bin", dtype=np.uint16, mode="r")
        val_data = np.memmap("/data/val.bin", dtype=np.uint16, mode="r")
        
        train_max = train_data.max() if len(train_data) > 0 else 0
        val_max = val_data.max() if len(val_data) > 0 else 0
        
        print(f"📊 Data validation:")
        print(f"   Train data: {len(train_data)} tokens, max_id: {train_max}")
        print(f"   Val data: {len(val_data)} tokens, max_id: {val_max}")
        
            
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
    
    print("✅ Data preparation completed!")

@app.local_entrypoint()
def main():

    # Prepare data first
    print("📁 Preparing data...")
    prepare_data.remote()

    result = train_sabiyarn.remote(
        attention_type="self_attention",
        dim=256,
        n_layers=10,
        n_heads=8,
        n_kv_heads=4,
        use_moe=False,
        n_routed_experts=4,
        n_activated_experts=2,
        use_multi_token_prediction=False,
        compile_model=True,
        use_cut_cross_entropy=False,
        layer_sharing=True,
        layer_sharing_strategy="immediate",
        n_unique_layers=5,
        max_iters=10000,  # Shorter for testing
        warmup_iters=300,
        lr_decay_iters=1000,
        wandb_run_name="small_standard_test",
        init_from="scratch",
   
    )
    
    if result:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
    
    return result

if __name__ == "__main__":
    main()
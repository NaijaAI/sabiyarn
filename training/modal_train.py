 #!/usr/bin/env python3
"""
Modal GPU training wrapper for SabiYarn models.
"""

import modal
from numpy._core.numeric import True_


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
    gpu="A10",  
    timeout=86400,  # 24 hours
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],  # Store W&B API key
    cpu=8,
    # memory=32768,  # 32GB RAM
)
def train_sabiyarn(volume: modal.Volume,):
    """
    Train SabiYarn model on Modal GPU with comprehensive monitoring.
    
    Args:
        All the configuration parameters for the training run.
    """
    import os
    import sys
    import yaml
    import shutil
    
    # Setup paths for Modal environment
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    # Import training modules
    from training.new_train import TrainingConfig, SabiYarnTrainer, AttentionType
    
    CONFIG_PATH = "/app/training/train_config.yaml"


    def clear_dedup_hash_folder(hash_dir: str, rehash: bool):
        """
        Delete the folder that stores the dedup hash registry
        to force a fresh deduplication run.

        Args:
            hash_dir (str): Path to the folder used by HashRegistry.
        """
        if os.path.exists(hash_dir) and rehash:
            shutil.rmtree(hash_dir)
            print(f"[INFO] Removed previous dedup hash folder: {hash_dir}")
        else:
            print(f"[INFO] No existing dedup hash folder found at: {hash_dir}")
            
    
    with open(CONFIG_PATH, "r") as f:
        conf = yaml.safe_load(f)
        
    
    clear_dedup_hash_folder(conf['hash']['registry_cache'], rehash=conf["data"]["rehash"])
        
    # Create configuration
    config = TrainingConfig(
        # Model Architecture
        attention_type=AttentionType(conf["model"]["attention_type"]),
        dim=conf['model']['dim'],
        n_layers=conf['model']['n_layers'],
        n_heads=conf['model']['n_heads'],
        n_kv_heads=conf['model']['n_kv_heads'],
        vocab_size=conf['model']['vocab_size'],
        max_seq_len=conf['model']['max_seq_len'],
        max_batch_size=conf['training']['max_batch_size'],
        tie_weights=conf['model']['tie_weights'],
        
        # MoE Configuration
        use_moe=conf['model']['use_moe'],
        n_routed_experts=conf['model']['n_routed_experts'],
        n_activated_experts=conf['model']['n_activated_experts'],
        moe_inter_dim=conf['model']['moe_inter_dim'],
        n_shared_experts=conf['model']['n_shared_experts'],
        score_function=conf['model']['score_function'],
        bias_update_speed=conf['model']['bias_update_speed'],
        moe_aux_loss_weight=conf['model']['moe_aux_loss_weight'],
        
        # Multi-Token Prediction
        use_multi_token_prediction=conf['model']['use_multi_token_prediction'],
        num_prediction_tokens=conf['model']['num_prediction_tokens'],
        mtp_only_training=conf['model']['mtp_only_training'],
        
        # Layer Sharing
        layer_sharing=conf['model']['layer_sharing'],
        layer_sharing_strategy=conf['model']['layer_sharing_strategy'],
        n_unique_layers=conf['model']['n_unique_layers'],
        
        #CCE
        use_cut_cross_entropy=conf['model']['use_cut_cross_entropy'],
        
        # Training Configuration
        train_batch_size=conf['training']['train_batch_size'],
        gradient_accumulation_steps=conf['training']['gradient_accumulation_steps'],
        learning_rate= conf['training']['learning_rate'],
        max_iters=conf['training']['max_iters'],
        weight_decay=conf['training']['weight_decay'],
        warmup_iters=conf['training']['warmup_iters'],
        lr_decay_iters=conf['training']['lr_decay_iters'],
        grad_clip=conf['training']['grad_clip'],
        optimizer_type=conf['training']['optimizer_type'],
        
        # Data paths (Modal persistent volume)
        dataset=conf['data']['datasets'],
        train_data_path=conf['data']['train_data_path'],
        eval_data_path=conf['data']['eval_data_path'],
        out_dir=conf['data']['out_dir'],
        eval_interval=conf['wandb']['eval_interval'],
        log_interval=conf['wandb']['log_interval'],
        # run_dir=conf['wandb']['run_dir'],
        
        # W&B Configuration
        wandb_log=conf["wandb"]["log"],
        wandb_project=conf['wandb']['project'],
        wandb_run_name=conf['wandb']['wandb_run_name'],
        wandb_tags=["Modal", "GPU", conf['model']['attention_type'],"SabiYarn"],
        
        # Enhanced monitoring for Modal
        log_grad_norm=conf['training']['log_grad_norm'],
        log_weights=True,
        log_system_metrics=True,
        log_moe_metrics=True,
        monitor_interval=50,
        
        init_from=conf['training']['init_from'],
        

        # System
        device=conf['training']['device'],
        dtype=conf['training']['dtype'],
        compile_model=conf['training']['compile_model'],  # Disable for debugging

        # Generation during training
        enable_generation_during_training=conf['training']['enable_generation_during_training'],
        
        # hashing registry for data deduplication
        hash_algo= conf['hash']['hash_algo'],
        registry_cache= conf['hash']['registry_cache'],
        map_size_gb= conf['hash']['map_size_gb'],
        overwrite_data= conf['data']['overwrite_data'], ## overwrite train.bin and val.bin if they exist
        use_custom_causal_mask=conf['training']['use_custom_causal_mask'],
    )
    
    print("Starting SabiYarn training on Modal GPU...")
    
    # Initialize and run trainer
    trainer = SabiYarnTrainer(config, volume)
    
    try:
        trainer.train()
        volume.commit()
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


@app.local_entrypoint()
def main():

    # Prepare data first
    # print("📁 Preparing data...")
    # prepare_data.remote()

    result = train_sabiyarn.remote(volume)
    
    if result:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
    
    return result

if __name__ == "__main__":
    main()
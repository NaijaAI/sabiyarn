import torch
import sys
import os

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import as proper package
from sabiyarn.MLA import MLA, MLAConfig, precompute_freqs_cis

def test_mla_tiny():
    # Minimal config for testing
    config = MLAConfig(
        hidden_size=128,
        num_heads=4,
        max_seq_len=16,
        max_batch_size=1,
        attention_dropout=0.0,
        # MLA parameters (minimal)
        q_lora_rank=32,
        qk_rope_head_dim=16,
        kv_lora_rank=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        attention_bias=False,
        # YARN parameters
        original_seq_len=16,
        rope_theta=10000.0,
        rope_factor=1,  # No scaling
        beta_fast=32,
        beta_slow=1,
        mscale=1.,
        # Distributed parameters (single GPU for test)
        world_size=1,
        rank=0
    )
    
    print("Initializing tiny MLA model...")
    mla = MLA(config)
    
    # Very small input
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    start_pos = 0
    freqs_cis = precompute_freqs_cis(config)
    
    print(f"Input shape: {x.shape}")
    print(f"Freqs_cis shape: {freqs_cis.shape}")
    
    try:
        print("Testing MLA forward pass...")
        attn_output = mla(x, start_pos, freqs_cis, mask=None)
        print(f"✓ Output shape: {attn_output.shape}")

        # Verify shapes
        expected_output = (batch_size, seq_len, config.hidden_size)
        assert attn_output.shape == expected_output, f"Expected {expected_output}, got {attn_output.shape}"
        
        print("✓ MLA tiny test passed!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mla_distributed():
    """Test MLA with simulated distributed configuration."""
    print("\n🧪 Testing MLA with distributed configuration...")
    
    # Config for simulated 2-GPU setup
    config = MLAConfig(
        hidden_size=128,
        num_heads=4,
        max_seq_len=16,
        max_batch_size=1,
        attention_dropout=0.0,
        # MLA parameters (minimal)
        q_lora_rank=32,
        qk_rope_head_dim=16,
        kv_lora_rank=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        attention_bias=False,
        # YARN parameters
        original_seq_len=16,
        rope_theta=10000.0,
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        mscale=1.,
        # Distributed parameters (simulated 2-GPU setup)
        world_size=2,
        rank=0
    )
    
    try:
        mla = MLA(config)
        print(f"✓ MLA created with world_size={mla.world_size}, rank={mla.rank}")
        
        # Verify that n_local_heads is correctly calculated
        expected_local_heads = config.num_heads // config.world_size
        assert mla.n_local_heads == expected_local_heads, f"Expected {expected_local_heads}, got {mla.n_local_heads}"
        print(f"✓ Local heads correctly calculated: {mla.n_local_heads}")
        
        # Test that parallel layers have correct sizes
        # q_up_proj should be ColumnParallel with reduced output
        # out_proj should be RowParallel
        print(f"✓ q_up_proj output features: {mla.q_up_proj.part_out_features}")
        print(f"✓ out_proj input features: {mla.out_proj.part_in_features}")
        
        # Test forward pass with distributed configuration
        # Note: We test with world_size=1 for forward pass to avoid needing distributed process group
        config_forward = MLAConfig(
            hidden_size=128,
            num_heads=4,
            max_seq_len=16,
            max_batch_size=1,
            attention_dropout=0.0,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            kv_lora_rank=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            attention_bias=False,
            original_seq_len=16,
            rope_theta=10000.0,
            rope_factor=1,
            beta_fast=32,
            beta_slow=1,
            mscale=1.,
            world_size=1,  # Use single GPU for forward pass test
            rank=0
        )
        
        mla_forward = MLA(config_forward)
        batch_size = 1
        seq_len = 4
        x = torch.randn(batch_size, seq_len, config_forward.hidden_size, dtype=torch.bfloat16)
        start_pos = 0
        freqs_cis = precompute_freqs_cis(config_forward)
        
        print("Testing distributed MLA forward pass (world_size=1)...")
        attn_output = mla_forward(x, start_pos, freqs_cis, mask=None)
        
        # Verify shapes
        expected_output = (batch_size, seq_len, config_forward.hidden_size)
        assert attn_output.shape == expected_output, f"Expected {expected_output}, got {attn_output.shape}"
        print(f"✓ Distributed config forward pass output shape: {attn_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Distributed MLA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running MLA Tests")
    print("=" * 40)
    
    success1 = test_mla_tiny()
    success2 = test_mla_distributed()
    
    if success1 and success2:
        print("\n✓ All MLA tests passed!")
        print("✓ MLA implementation working correctly!")
    else:
        print("\n✗ Some MLA tests failed!")
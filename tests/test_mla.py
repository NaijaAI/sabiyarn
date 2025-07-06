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
        mscale=1.
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

if __name__ == "__main__":
    success = test_mla_tiny()
    if success:
        print("✓ MLA implementation working correctly!")
    else:
        print("✗ MLA test failed!")
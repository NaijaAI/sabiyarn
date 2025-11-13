import torch
import sys
import os

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import as proper package
from sabiyarn.differential_attention import DiffAttention, DiffAttnArgs, precompute_freqs_cis

def test_diff_attention():
    # Create test config
    args = DiffAttnArgs(
        depth=3,
        max_batch_size=2,
        n_heads=8,  # This is half of transformer heads
        embed_dim=512,
        n_kv_heads=None,
        max_seq_len=64,
        norm_eps=1e-5
    )
    
    # Initialize model
    model = DiffAttention(args)
    
    # Create test input
    batch_size = 1
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.embed_dim)
    
    # Create frequency tensor
    freqs_cis = precompute_freqs_cis(args.embed_dim // args.n_heads // 2, seq_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Freqs_cis shape: {freqs_cis.shape}")
    print(f"Lambda init: {model.lambda_init}")
    
    try:
        # Test without mask (should apply causal masking)
        print("Testing with attn_mask=None (automatic causal masking)...")
        output_no_mask = model(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
        print(f"✓ Auto-causal output shape: {output_no_mask.shape}")
        
        # Test with explicit mask
        print("Testing with explicit causal mask...")
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        output_explicit_mask = model(x, start_pos=0, freqs_cis=freqs_cis, mask=causal_mask)
        print(f"✓ Explicit-mask output shape: {output_explicit_mask.shape}")
        
        # Verify output shape is correct
        expected_shape = (batch_size, seq_len, args.embed_dim)
        assert output_no_mask.shape == expected_shape, f"Expected {expected_shape}, got {output_no_mask.shape}"
        
        print(f"✓ Expected output shape: {expected_shape}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diff_attention()
    if success:
        print("✓ Differential Attention test passed!")
    else:
        print("✗ Differential Attention test failed!")
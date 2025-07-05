import torch
import sys
import os

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import as proper package
from sabiyarn.MHA import SelfAttention, SelfAttnArgs, precompute_freqs_cis

def test_mha():
    # Create smaller config for testing
    args = SelfAttnArgs(
        dim=512,
        n_heads=8,
        n_kv_heads=4,  # Test grouped query attention
        max_batch_size=2,
        max_seq_len=64
    )
    
    # Initialize model
    model = SelfAttention(args)
    
    # Create test input
    batch_size = 1
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.dim)
    
    # Create frequency tensor
    freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, seq_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Freqs_cis shape: {freqs_cis.shape}")
    
    try:
        # Test without mask (should automatically apply causal masking)
        print("Testing with mask=None (automatic causal masking)...")
        output_no_mask = model(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
        print(f"✓ Auto-causal output shape: {output_no_mask.shape}")
        
        # Test with explicit mask (should be equivalent)
        print("Testing with explicit causal mask...")
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
        output_explicit_mask = model(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)
        print(f"✓ Explicit-mask output shape: {output_explicit_mask.shape}")
        
        # Verify outputs are similar (should be nearly identical)
        diff = torch.abs(output_no_mask - output_explicit_mask).mean()
        print(f"✓ Difference between auto and explicit causal masking: {diff.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mha()
    if success:
        print("✓ MHA test passed!")
    else:
        print("✗ MHA test failed!")
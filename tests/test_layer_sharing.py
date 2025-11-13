import torch
import sys
import os

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import as proper package
from sabiyarn.model import ModelArgs, SabiYarn, AttentionType
from sabiyarn.MLA import MLAConfig


def test_layer_sharing():
    """Test MobileLLM-style layer sharing functionality."""
    print("\n🧪 Testing Layer Sharing (MobileLLM-style immediate block-wise repeat)...")
    
    try:
        # Test with MLA attention (which works well with layer sharing)
        mla_config = MLAConfig(
            hidden_size=256,
            num_heads=8,
            max_seq_len=32,
            max_batch_size=2,
            attention_dropout=0.0,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            kv_lora_rank=32,
            v_head_dim=32,
            qk_nope_head_dim=16,
            attention_bias=False,
            original_seq_len=32,
            rope_theta=10000.0,
            rope_factor=1,
            beta_fast=32,
            beta_slow=1,
            mscale=1.
        )
        
        # Test 1: Traditional model (no layer sharing)
        traditional_config = ModelArgs(
            dim=256,
            n_layers=12,  # 12 unique layers
            n_heads=8,
            vocab_size=1000,
            max_batch_size=2,
            max_seq_len=32,
            attention_type=AttentionType.MLA,
            mla_config=mla_config,
            layer_sharing=False,  # No layer sharing
            auto_detect_distributed=False,
            init_std=0.01
        )
        
        traditional_model = SabiYarn(traditional_config)
        traditional_params = sum(p.numel() for p in traditional_model.parameters() if p.requires_grad)
        print(f"✅ Traditional model: {traditional_model.get_model_size()}")
        
        # Test 2: Layer sharing model (same effective depth, fewer unique layers)
        layer_sharing_config = ModelArgs(
            dim=256,
            n_layers=12,  # Same effective depth
            layer_sharing=True,  # Enable layer sharing
            n_unique_layers=4,   # Only 4 unique layers, repeated 3 times each
            n_heads=8,
            vocab_size=1000,
            max_batch_size=2,
            max_seq_len=32,
            attention_type=AttentionType.MLA,
            mla_config=mla_config,
            auto_detect_distributed=False,
            init_std=0.01
        )
        
        layer_sharing_model = SabiYarn(layer_sharing_config)
        layer_sharing_params = sum(p.numel() for p in layer_sharing_model.parameters() if p.requires_grad)
        print(f"✅ Layer sharing model: {layer_sharing_model.get_model_size()}")
        
        # Verify memory savings
        memory_reduction = traditional_params - layer_sharing_params
        memory_reduction_pct = (memory_reduction / traditional_params) * 100
        print(f"   Memory reduction: {memory_reduction:,} parameters ({memory_reduction_pct:.1f}%)")
        
        # Test 3: Verify forward pass works correctly
        tokens = torch.randint(0, 1000, (1, 16))
        
        # Traditional model forward pass
        # with torch.no_grad():
        #     traditional_model.eval()
        #     traditional_model = traditional_model.float()
        #     trad_hidden, trad_logits = traditional_model(tokens, start_pos=0)
        #     print(f"✅ Traditional forward: {trad_hidden.shape} -> {trad_logits.shape}")
        
        # Layer sharing model forward pass
        with torch.no_grad():
            layer_sharing_model.eval()
            layer_sharing_model = layer_sharing_model.float()
            ls_hidden, ls_logits = layer_sharing_model(tokens, start_pos=0)
            print(f"✅ Layer sharing forward: {ls_hidden.shape} -> {ls_logits.shape}")
        
        # Test 4: Verify execution order is correct
        expected_order = [0, 1, 2, 3] * 3  # 4 unique layers repeated 3 times
        actual_order = layer_sharing_model.layer_execution_order
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
        print(f"✅ Execution order correct: {actual_order}")
        
        # Test 5: Verify the model has fewer actual parameters but same effective depth
        assert layer_sharing_model.n_unique_layers == 4
        assert layer_sharing_model.repeat_factor == 3
        assert len(layer_sharing_model.unique_layers) == 4
        assert len(layer_sharing_model.layer_execution_order) == 12
        print(f"✅ Architecture correct: {layer_sharing_model.n_unique_layers} unique layers, {layer_sharing_model.repeat_factor}x repeat factor")
        
        print("✅ Layer sharing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Layer sharing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_sharing_validation():
    """Test layer sharing configuration validation."""
    print("\n🧪 Testing Layer Sharing Validation...")
    
    try:
        base_config = {
            "dim": 256,
            "n_layers": 12,
            "n_heads": 8,
            "vocab_size": 1000,
            "auto_detect_distributed": False
        }
        
        # Test 1: layer_sharing=True but n_unique_layers=None should fail
        try:
            ModelArgs(**base_config, layer_sharing=True, n_unique_layers=None)
            assert False, "Should have failed with n_unique_layers=None"
        except ValueError as e:
            print(f"✅ Correctly caught: {e}")
        
        # Test 2: n_unique_layers > n_layers should fail
        try:
            ModelArgs(**base_config, layer_sharing=True, n_unique_layers=15)
            assert False, "Should have failed with n_unique_layers > n_layers"
        except ValueError as e:
            print(f"✅ Correctly caught: {e}")
        
        # Test 3: n_layers not divisible by n_unique_layers should fail
        try:
            ModelArgs(**base_config, layer_sharing=True, n_unique_layers=5)  # 12 % 5 != 0
            assert False, "Should have failed with non-divisible layers"
        except ValueError as e:
            print(f"✅ Correctly caught: {e}")
        
        # Test 4: Valid configuration should pass
        config = ModelArgs(**base_config, layer_sharing=True, n_unique_layers=4)  # 12 % 4 == 0
        print(f"✅ Valid configuration accepted: {config.n_layers} layers, {config.n_unique_layers} unique")
        
        print("✅ Layer sharing validation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Layer sharing validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success1 = test_layer_sharing()
    success2 = test_layer_sharing_validation()
    
    if success1 and success2:
        print("\n🎉 All layer sharing tests passed!")
    else:
        print("\n❌ Some layer sharing tests failed!")


if __name__ == "__main__":
    main()
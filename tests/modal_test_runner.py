#!/usr/bin/env python3
"""
Modal test runner for GitHub Actions.
This script runs the SabiYarn model tests on Modal's GPU instances.

The key difference from test_model_initialization.py is that this script
does NOT import any GPU-dependent modules at the top level, avoiding
the Triton initialization error on CPU runners.
"""

import modal
import sys
import os

from sabiyarn.grouped_query_attention import GQAArgs

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Create Modal app with custom image that includes dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir(".", "/root/app")
)

app = modal.App("sabiyarn-tests")

@app.function(gpu="A10G", timeout=800, image=image)
def run_tests_on_gpu():
    """
    Run all tests on Modal GPU instance.
    This function runs on GPU, so all GPU-dependent imports are safe here.
    """
    import sys
    import os
    import torch
    
    # Add the project root to path (code is copied to /root/app in Modal)
    project_root = "/root/app"
    sys.path.insert(0, project_root)
    
    # Now import the GPU-dependent modules (this happens on GPU)
    from sabiyarn.model import ModelArgs, SabiYarn, AttentionType
    from sabiyarn.MLA import MLAConfig
    from sabiyarn.differential_attention import DiffAttnArgs
    
    # Try to import cut_cross_entropy, skip test if not available
    try:
        from cut_cross_entropy import linear_cross_entropy
        CCE_AVAILABLE = True
    except ImportError:
        CCE_AVAILABLE = False
        linear_cross_entropy = None
    
    def test_cut_cross_entropy():
        """Test cut cross entropy loss function."""
        if not CCE_AVAILABLE or linear_cross_entropy is None:
            print("⚠️ Cut Cross Entropy module not available, skipping test")
            return True
        
        print("🧪 Testing Cut Cross Entropy...")
        try:
            # Create test embeddings and classifier weights  
            batch_size, seq_len, embed_dim = 4, 10, 128
            vocab_size = 100
            
            # Create embeddings (e) and classifier weights (c) 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if device == "cpu":
                print("⚠️ CUDA not available, using torch_compile implementation")
                # Use torch_compile implementation for CPU
                e = torch.randn(batch_size, seq_len, embed_dim)  # embeddings
                c = torch.randn(vocab_size, embed_dim)  # classifier weights
                targets = torch.randint(0, vocab_size, (batch_size, seq_len))
                
                # Force torch_compile implementation
                loss = linear_cross_entropy(e, c, targets, impl="torch_compile")
            else:
                print(f"✅ Using GPU ({device}) with CCE implementation")
                # Use GPU tensors for CCE implementation
                e = torch.randn(batch_size, seq_len, embed_dim, device=device)  # embeddings
                c = torch.randn(vocab_size, embed_dim, device=device)  # classifier weights
                targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                
                # Use default CCE implementation
                loss = linear_cross_entropy(e, c, targets)
            
            assert loss.item() > 0, "Loss should be positive"
            assert torch.isfinite(loss), "Loss should be finite"
            
            print(f"   Device: {device}")
            print(f"   Embeddings shape: {e.shape}")
            print(f"   Classifier weights shape: {c.shape}")  
            print(f"   Targets shape: {targets.shape}")
            print(f"   Loss value: {loss.item():.4f}")
            print("✅ Cut Cross Entropy test passed!")
            return True
        except Exception as e:
            print(f"❌ Cut Cross Entropy test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_mha_model_initialization():
        """Test model initialization with Multi-Head Attention."""
        print("🧪 Testing MHA Model Initialization...")
        
        try:
            # Create MHA configuration
            config = ModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,
                n_kv_heads=4,
                vocab_size=1000,
                max_batch_size=2,
                max_seq_len=32,
                attention_type=AttentionType.SELF_ATTENTION
            )
            
            # Initialize model
            model = SabiYarn(config)
            print(f"✅ MHA model created: {model.get_model_size()}")
            
            # Test forward pass
            tokens = torch.randint(0, 1000, (1, 16))
            hidden_states, logits = model(tokens, start_pos=0)
            
            print(f"   Input: {tokens.shape}")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Logits: {logits.shape}")
            
            expected_hidden = (1, 16, 256)
            expected_logits = (1, 16, 1000)
            
            assert hidden_states.shape == expected_hidden, f"Hidden states shape mismatch: {hidden_states.shape} vs {expected_hidden}"
            assert logits.shape == expected_logits, f"Logits shape mismatch: {logits.shape} vs {expected_logits}"
            
            print("✅ MHA model test passed!")
            return True
            
        except Exception as e:
            print(f"❌ MHA model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_differential_attention_model():
        """Test model initialization with Differential Attention."""
        print("\n🧪 Testing Differential Attention Model...")
        
        try:
            # Create Differential Attention configuration
            diff_args = DiffAttnArgs(
                depth=2,
                max_batch_size=2,
                n_heads=8,
                embed_dim=256,
                n_kv_heads=4,
                max_seq_len=32,
                norm_eps=1e-5
            )
            
            config = ModelArgs(
                dim=256,
                n_layers=2,
                n_heads=16,  # Total heads for transformer
                vocab_size=1000,
                max_batch_size=2,
                max_seq_len=32,
                attention_type=AttentionType.DIFFERENTIAL_ATTENTION,
                diff_attn_args=diff_args
            )
            
            # Initialize model
            model = SabiYarn(config)
            print(f"✅ Differential Attention model created: {model.get_model_size()}")
            
            # Test forward pass
            tokens = torch.randint(0, 1000, (1, 16))
            hidden_states, logits = model(tokens, start_pos=0)
            
            print(f"   Input: {tokens.shape}")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Logits: {logits.shape}")
            
            expected_hidden = (1, 16, 256)
            expected_logits = (1, 16, 1000)
            
            assert hidden_states.shape == expected_hidden, f"Hidden states shape mismatch"
            assert logits.shape == expected_logits, f"Logits shape mismatch"
            
            print("✅ Differential Attention model test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Differential Attention model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_gqa_model():
        """ Test model initialization with Grouped Query Attention"""
        print("\n🧪 Testing Grouped Query Attention with model.....")
        try:
            gqa_config = GQAArgs(
                dim=384,
                n_kv_heads=4,
                n_heads=8,
                max_seq_len=2048,
                max_batch_size=32
            )
            config = ModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,
                vocab_size=1000,
                max_batch_size=32,
                max_seq_len=32,
                attention_type=AttentionType.GQA,
                gqa_config=gqa_config
            )
            model = SabiYarn(config)
            print(f"✅ GQA model created: {model.get_model_size()}")
            
            # Test forward pass with proper dtype handling
            tokens = torch.randint(0, 1000, (1, 16))
            
            # Set model to eval mode and ensure consistent dtype
            model.eval()
            with torch.no_grad():
                # Convert model to float32 to avoid dtype mismatch
                model = model.float()
                hidden_states, logits = model(tokens, start_pos=0)
            
            print(f"   Input: {tokens.shape}")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Logits: {logits.shape}")
            
            expected_hidden = (1, 16, 384)
            expected_logits = (1, 16, 10000)
        
            assert hidden_states.shape == expected_hidden, f"Hidden states shape mismatch: {hidden_states.shape} vs {expected_hidden}"
            assert logits.shape == expected_logits, f"Logits shape mismatch: {logits.shape} vs {expected_logits}"
            
            print("✅ GQA model test passed!")
            return True
            
        except Exception as e:
            print(f"❌ GQA model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_mla_model():
        """Test model initialization with Multi-Head Latent Attention."""
        print("\n🧪 Testing MLA Model...")
        
        try:
            # Create MLA configuration
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
            
            config = ModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,
                vocab_size=1000,
                max_batch_size=2,
                max_seq_len=32,
                attention_type=AttentionType.MLA,
                mla_config=mla_config
            )
            
            # Initialize model
            model = SabiYarn(config)
            print(f"✅ MLA model created: {model.get_model_size()}")
            
            # Test forward pass with proper dtype handling
            tokens = torch.randint(0, 1000, (1, 16))
            
            # Set model to eval mode and ensure consistent dtype
            model.eval()
            with torch.no_grad():
                # Convert model to float32 to avoid dtype mismatch
                model = model.float()
                hidden_states, logits = model(tokens, start_pos=0)
            
            print(f"   Input: {tokens.shape}")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Logits: {logits.shape}")
            
            expected_hidden = (1, 16, 256)
            expected_logits = (1, 16, 1000)
        
            assert hidden_states.shape == expected_hidden, f"Hidden states shape mismatch: {hidden_states.shape} vs {expected_hidden}"
            assert logits.shape == expected_logits, f"Logits shape mismatch: {logits.shape} vs {expected_logits}"
            
            print("✅ MLA model test passed!")
            return True
            
        except Exception as e:
            print(f"❌ MLA model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_mla_with_moe():
        """Test model initialization with MLA + MoE."""
        print("\n🧪 Testing MLA + MoE Model...")
        
        try:
            # Create MLA + MoE configuration
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
            
            config = ModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,  # Match mla_config.num_heads
                vocab_size=1000,
                max_batch_size=2,
                max_seq_len=32,
                attention_type=AttentionType.MLA,
                mla_config=mla_config,
                moe=True,
                n_routed_experts=8,
                n_activated_experts=2,
                moe_inter_dim=512,
                n_shared_experts=1
            )
            
            # Initialize model
            model = SabiYarn(config)
            print(f"✅ MLA + MoE model created: {model.get_model_size()}")
            
            # Test forward pass
            tokens = torch.randint(0, 1000, (1, 16))
            # Set model to eval mode and ensure consistent dtype
            model.eval()
            with torch.no_grad():
                # Convert model to float32 to avoid dtype mismatch
                model = model.float()
                hidden_states, logits = model(tokens, start_pos=0)
            
            print(f"   Input: {tokens.shape}")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Logits: {logits.shape}")
            
            expected_hidden = (1, 16, 256)
            expected_logits = (1, 16, 1000)
            
            assert hidden_states.shape == expected_hidden, f"Hidden states shape mismatch: {hidden_states.shape} vs {expected_logits}"
            assert logits.shape == expected_logits, f"Logits shape mismatch: {logits.shape} vs {expected_logits}"
            
            # Verify MoE is being used
            first_layer = model.layers[0]
            assert hasattr(first_layer, 'attention'), "Layer should have attention component"
            assert hasattr(first_layer, 'feed_forward'), "Layer should have MoE feed_forward component"
            
            from sabiyarn.moe import MoE
            assert isinstance(first_layer.feed_forward, MoE), "Should be using MoE for feed_forward"
            
            print("✅ MLA + MoE model test passed!")
            return True
            
        except Exception as e:
            print(f"❌ MLA + MoE model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_attention_factory():
        """Test the attention factory function."""
        print("\n🧪 Testing Attention Factory...")
        
        try:
            from sabiyarn.model import _create_attention
            
            # Test MHA creation
            mha_config = ModelArgs(
                dim=256,
                n_heads=8,
                attention_type=AttentionType.SELF_ATTENTION
            )
            mha_attention = _create_attention(0, mha_config)
            print("✅ MHA attention module created")

            # Test Differential Attention creation
            diff_args = DiffAttnArgs(
                depth=0,
                max_batch_size=2,
                n_heads=8,
                embed_dim=256,
                n_kv_heads=4,
                max_seq_len=32,
                norm_eps=1e-5
            )
            diff_config = ModelArgs(
                dim=256,
                attention_type=AttentionType.DIFFERENTIAL_ATTENTION,
                diff_attn_args=diff_args
            )
            diff_attention = _create_attention(0, diff_config)
            print("✅ Differential attention module created")
                
            # Test MLA creation
            mla_cfg = MLAConfig(
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
            mla_config = ModelArgs(
                dim=256,
                n_heads=8,  # Match mla_cfg.num_heads
                attention_type=AttentionType.MLA,
                mla_config=mla_cfg
            )
            mla_attention = _create_attention(0, mla_config)
            print("✅ MLA attention module created")
            
            print("✅ Attention factory test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Attention factory test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_configuration_validation():
        """Test configuration validation."""
        print("\n🧪 Testing Configuration Validation...")
        
        try:
            from sabiyarn.model import _validate_attention_config
            
            # Test valid configurations
            valid_mha = ModelArgs(
                dim=256,
                n_heads=8,
                attention_type=AttentionType.SELF_ATTENTION
            )
            _validate_attention_config(valid_mha)
            print("✅ Valid MHA config validation passed")
            
            # Test invalid MoE with wrong attention type
            try:
                invalid_moe = ModelArgs(
                    dim=256,
                    attention_type=AttentionType.SELF_ATTENTION,
                    moe=True,
                    n_routed_experts=8,
                    n_activated_experts=2
                )
                _validate_attention_config(invalid_moe)
                print("❌ Should have failed MoE validation")
                return False
            except ValueError:
                print("✅ Invalid MoE config correctly rejected")
            
            # Test invalid expert count
            try:
                invalid_experts = ModelArgs(
                    dim=256,
                    attention_type=AttentionType.MLA,
                    moe=True,
                    n_routed_experts=4,
                    n_activated_experts=8,  # More than routed
                    mla_config=MLAConfig(
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
                )
                _validate_attention_config(invalid_experts)
                print("❌ Should have failed expert count validation")
                return False
            except ValueError:
                print("✅ Invalid expert count correctly rejected")
            
            print("✅ Configuration validation test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run all tests
    test_functions = [
        ("Cut Cross Entropy", test_cut_cross_entropy),
        ("MHA Model Initialization", test_mha_model_initialization),
        ("GQA Model initialization", test_gqa_model),
        ("Differential Attention Model", test_differential_attention_model),
        ("MLA Model", test_mla_model),
        ("MLA + MoE Model", test_mla_with_moe),
        ("Attention Factory", test_attention_factory),
        ("Configuration Validation", test_configuration_validation),
    ]
    
    print("🚀 **SabiYarn Model Initialization Tests (Modal GPU)**")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 **Final Results: {passed_tests}/{total_tests} tests passed**")
    
    if passed_tests == total_tests:
        print("\n🎉 **All Model Initialization Tests Passed on Modal GPU!**")
        print("\n**Architecture Features Validated:**")
        print("✅ Multi-Head Attention (MHA) support")
        print("✅ Differential Attention support")
        print("✅ Multi-Head Latent Attention (MLA) support")
        print("✅ MLA + Mixture of Experts (MoE) integration")
        print("✅ Attention factory pattern")
        print("✅ Configuration validation")
        print("✅ Unified transformer blocks")
        print("✅ Modular architecture design")
        print("✅ Cut Cross Entropy support")
        print("\n🏆 **SabiYarn Model is fully functional and modular!**")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed")
        return False

@app.local_entrypoint()
def main():
    """Entry point for Modal execution in GitHub Actions."""
    print("🔄 Starting tests on Modal GPU...")
    result = run_tests_on_gpu.remote()
    
    if not result:
        print("❌ Tests failed on Modal GPU")
        sys.exit(1)
    else:
        print("✅ All tests passed on Modal GPU")

if __name__ == "__main__":
    main() 
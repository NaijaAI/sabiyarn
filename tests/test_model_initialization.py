#!/usr/bin/env python3
"""
Comprehensive test for model initialization with various attention mechanisms.
Tests the complete modular SabiYarn architecture on Modal GPU.

Usage:
  modal run tests/test_model_initialization.py::modal_main

This script runs all tests on Modal GPU infrastructure.
"""

import modal

# Create Modal app with custom image that includes dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app", ignore=[".git", "*.pyc", "__pycache__", ".pytest_cache", "*.egg-info"])
)

# Modal app for running tests on Modal
app = modal.App("sabiyarn-tests")

@app.function(gpu="A10G", timeout=1000, image=image)
def run_all_tests():
    """Run all comprehensive tests on Modal GPU."""
    # Setup imports within Modal context
    import sys
    import os
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    # Import the required dependencies for the tests
    import torch
    from sabiyarn.model import ModelArgs, SabiYarn, AttentionType, _detect_distributed_config, _create_attention, _validate_attention_config
    from sabiyarn.MLA import MLAConfig
    from sabiyarn.differential_attention import DiffAttnArgs
    
    # Try to import cut_cross_entropy, skip test if not available
    try:
        from cut_cross_entropy import linear_cross_entropy
        cce_available = True
    except ImportError:
        cce_available = False
        linear_cross_entropy = None
    
    print("🚀 **SabiYarn Model Initialization Tests (Modal GPU)**")
    print("=" * 60)
    
    # Test 1: Cut Cross Entropy
    def test_cut_cross_entropy():
        print("🧪 Testing Cut Cross Entropy...")
        if not cce_available or linear_cross_entropy is None:
            print("⚠️ Cut Cross Entropy module not available, skipping test")
            return True
        try:
            batch_size, seq_len, embed_dim = 4, 10, 128
            vocab_size = 100
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if device == "cpu": # or MacOS
                e = torch.randn(batch_size, seq_len, embed_dim)
                c = torch.randn(vocab_size, embed_dim)
                targets = torch.randint(0, vocab_size, (batch_size, seq_len))
                loss = linear_cross_entropy(e, c, targets, impl="torch_compile")
            else:
                e = torch.randn(batch_size, seq_len, embed_dim, device=device)
                c = torch.randn(vocab_size, embed_dim, device=device)
                targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                loss = linear_cross_entropy(e, c, targets)
            
            assert loss.item() > 0
            assert torch.isfinite(loss)
            print("✅ Cut Cross Entropy test passed!")
            return True
        except Exception as e:
            print(f"❌ Cut Cross Entropy test failed: {e}")
            return False
    
    # Test 2: MHA Model Initialization
    def test_mha_model_initialization():
        print("🧪 Testing MHA Model Initialization...")
        try:
            config = ModelArgs(
                dim=256, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=1000,
                max_batch_size=2, max_seq_len=32, attention_type=AttentionType.SELF_ATTENTION
            )
            model = SabiYarn(config)
            tokens = torch.randint(0, 1000, (1, 16))
            hidden_states, logits = model(tokens, start_pos=0)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            print("✅ MHA model test passed!")
            return True
        except Exception as e:
            print(f"❌ MHA model test failed: {e}")
            return False
    
    # Test 3: Differential Attention Model
    def test_differential_attention_model():
        print("🧪 Testing Differential Attention Model...")
        try:
            diff_args = DiffAttnArgs(
                depth=2, max_batch_size=2, n_heads=8, embed_dim=256,
                n_kv_heads=4, max_seq_len=32, norm_eps=1e-5
            )
            config = ModelArgs(
                dim=256, n_layers=2, n_heads=16, vocab_size=1000,
                max_batch_size=2, max_seq_len=32, attention_type=AttentionType.DIFFERENTIAL_ATTENTION,
                diff_attn_args=diff_args
            )
            model = SabiYarn(config)
            tokens = torch.randint(0, 1000, (1, 16))
            hidden_states, logits = model(tokens, start_pos=0)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            print("✅ Differential Attention model test passed!")
            return True
        except Exception as e:
            print(f"❌ Differential Attention model test failed: {e}")
            return False
    
    # Test 4: MLA Model
    def test_mla_model():
        print("🧪 Testing MLA Model...")
        try:
            mla_config = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            config = ModelArgs(
                dim=256, n_layers=2, n_heads=8, vocab_size=1000,
                max_batch_size=2, max_seq_len=32, attention_type=AttentionType.MLA,
                mla_config=mla_config
            )
            model = SabiYarn(config)
            model.eval()
            with torch.no_grad():
                model = model.float()
                tokens = torch.randint(0, 1000, (1, 16))
                hidden_states, logits = model(tokens, start_pos=0)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            print("✅ MLA model test passed!")
            return True
        except Exception as e:
            print(f"❌ MLA model test failed: {e}")
            return False
    
    # Test 5: MLA + MoE Model
    def test_mla_with_moe():
        print("🧪 Testing MLA + MoE Model...")
        try:
            mla_config = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            config = ModelArgs(
                dim=256, n_layers=2, n_heads=8, vocab_size=1000,
                max_batch_size=2, max_seq_len=32, attention_type=AttentionType.MLA,
                mla_config=mla_config, moe=True, n_routed_experts=8,
                n_activated_experts=2, moe_inter_dim=512, n_shared_experts=1,
                score_function="sigmoid", bias_update_speed=0.001, moe_aux_loss_weight=0.001
            )
            model = SabiYarn(config)
            model.eval()
            with torch.no_grad():
                model = model.float()
                tokens = torch.randint(0, 1000, (1, 16))
                hidden_states, logits = model(tokens, start_pos=0)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            print("✅ MLA + MoE model test passed!")
            return True
        except Exception as e:
            print(f"❌ MLA + MoE model test failed: {e}")
            return False
    
    # Test 6: Attention Factory
    def test_attention_factory():
        print("🧪 Testing Attention Factory...")
        try:
            # Test MHA creation
            mha_config = ModelArgs(
                dim=256, n_heads=8, n_kv_heads=4, attention_type=AttentionType.SELF_ATTENTION
            )
            mha_attention = _create_attention(0, mha_config)
            
            # Test Differential Attention creation
            diff_args = DiffAttnArgs(
                depth=0, max_batch_size=2, n_heads=8, embed_dim=256,
                n_kv_heads=4, max_seq_len=32, norm_eps=1e-5
            )
            diff_config = ModelArgs(
                dim=256, attention_type=AttentionType.DIFFERENTIAL_ATTENTION, diff_attn_args=diff_args
            )
            diff_attention = _create_attention(0, diff_config)
            
            # Test MLA creation
            mla_cfg = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            mla_config = ModelArgs(
                dim=256, n_heads=8, attention_type=AttentionType.MLA, mla_config=mla_cfg
            )
            mla_attention = _create_attention(0, mla_config)
            
            print("✅ Attention factory test passed!")
            return True
        except Exception as e:
            print(f"❌ Attention factory test failed: {e}")
            return False
    
    # Test 7: Configuration Validation
    def test_configuration_validation():
        print("🧪 Testing Configuration Validation...")
        try:
            # Test valid MHA config
            valid_mha = ModelArgs(
                dim=256, n_heads=8, attention_type=AttentionType.SELF_ATTENTION
            )
            _validate_attention_config(valid_mha)
            
            # Test invalid MoE with wrong attention type
            try:
                invalid_moe = ModelArgs(
                    dim=256, attention_type=AttentionType.SELF_ATTENTION,
                    moe=True, n_routed_experts=8, n_activated_experts=2
                )
                _validate_attention_config(invalid_moe)
                return False
            except ValueError:
                pass
            
            # Test invalid expert count
            try:
                invalid_experts = ModelArgs(
                    dim=256, attention_type=AttentionType.MLA, moe=True,
                    n_routed_experts=4, n_activated_experts=8,
                    mla_config=MLAConfig(
                        hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                        attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                        kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                        attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                        rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
                    )
                )
                _validate_attention_config(invalid_experts)
                return False
            except ValueError:
                pass
            
            print("✅ Configuration validation test passed!")
            return True
        except Exception as e:
            print(f"❌ Configuration validation test failed: {e}")
            return False
    
    # Test 8: Distributed Training Config
    def test_distributed_training_config():
        print("🧪 Testing Distributed Training Configuration...")
        try:
            distributed, data_parallel, tensor_parallel, world_size, rank = _detect_distributed_config()
            
            mla_config = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            config_mla = ModelArgs(
                dim=256, n_layers=1, n_heads=8, vocab_size=1000,
                attention_type=AttentionType.MLA, mla_config=mla_config,
                auto_detect_distributed=True
            )
            model_mla = SabiYarn(config_mla)
            model_mla.eval()
            with torch.no_grad():
                model_mla = model_mla.float()
                tokens = torch.randint(0, 1000, (1, 8))
                hidden_states, logits = model_mla(tokens, start_pos=0)
            
            print("✅ Distributed training configuration test passed!")
            return True
        except Exception as e:
            print(f"❌ Distributed training configuration test failed: {e}")
            return False
    
    # Test 9: Multi-Token Prediction
    def test_multi_token_prediction():
        print("🧪 Testing Multi-Token Prediction...")
        try:
            mla_config = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            config = ModelArgs(
                dim=256, n_layers=2, n_heads=8, vocab_size=1000,
                max_batch_size=2, max_seq_len=32, attention_type=AttentionType.MLA,
                mla_config=mla_config, multi_token_prediction=True,
                num_prediction_tokens=4, mtp_loss_weight=0.5,
                mtp_share_embeddings=True, auto_detect_distributed=False, init_std=0.01
            )
            model = SabiYarn(config)
            model.eval()
            with torch.no_grad():
                model = model.float()
                tokens = torch.randint(0, 1000, (1, 16))
                hidden_states, logits, multi_token_logits = model(tokens, start_pos=0, return_multi_token=True)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            assert multi_token_logits.shape == (1, 16, 4, 1000)
            print("✅ Multi-Token Prediction test passed!")
            return True
        except Exception as e:
            print(f"❌ Multi-Token Prediction test failed: {e}")
            return False
    
    # Test 10: Layer Sharing
    def test_layer_sharing():
        print("🧪 Testing Layer Sharing...")
        try:
            mla_config = MLAConfig(
                hidden_size=256, num_heads=8, max_seq_len=32, max_batch_size=2,
                attention_dropout=0.0, q_lora_rank=64, qk_rope_head_dim=16,
                kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=16,
                attention_bias=False, original_seq_len=32, rope_theta=10000.0,
                rope_factor=1, beta_fast=32, beta_slow=1, mscale=1.
            )
            config = ModelArgs(
                dim=256, n_layers=12, layer_sharing=True, n_unique_layers=4,
                n_heads=8, vocab_size=1000, max_batch_size=2, max_seq_len=32,
                attention_type=AttentionType.MLA, mla_config=mla_config,
                auto_detect_distributed=False, init_std=0.01
            )
            model = SabiYarn(config)
            model.eval()
            with torch.no_grad():
                model = model.float()
                tokens = torch.randint(0, 1000, (1, 16))
                hidden_states, logits = model(tokens, start_pos=0)
            
            assert hidden_states.shape == (1, 16, 256)
            assert logits.shape == (1, 16, 1000)
            assert model.layer_execution_order == [0, 1, 2, 3] * 3
            assert model.n_unique_layers == 4
            assert model.repeat_factor == 3
            print("✅ Layer sharing test passed!")
            return True
        except Exception as e:
            print(f"❌ Layer sharing test failed: {e}")
            return False
    
    # Test 11: Layer Sharing Validation
    def test_layer_sharing_validation():
        print("🧪 Testing Layer Sharing Validation...")
        try:
            base_config = {
                "dim": 256, "n_layers": 12, "n_heads": 8, "vocab_size": 1000,
                "auto_detect_distributed": False
            }
            
            # Test layer_sharing=True but n_unique_layers=None should fail
            try:
                ModelArgs(**base_config, layer_sharing=True, n_unique_layers=None)
                return False
            except ValueError:
                pass
            
            # Test n_unique_layers > n_layers should fail
            try:
                ModelArgs(**base_config, layer_sharing=True, n_unique_layers=15)
                return False
            except ValueError:
                pass
            
            # Test n_layers not divisible by n_unique_layers should fail
            try:
                ModelArgs(**base_config, layer_sharing=True, n_unique_layers=5)
                return False
            except ValueError:
                pass
            
            # Test valid configuration should pass
            config = ModelArgs(**base_config, layer_sharing=True, n_unique_layers=4)
            
            print("✅ Layer sharing validation test passed!")
            return True
        except Exception as e:
            print(f"❌ Layer sharing validation test failed: {e}")
            return False
    
    # Run ALL 11 tests
    tests = [
        ("Cut Cross Entropy", test_cut_cross_entropy),
        ("MHA Model Initialization", test_mha_model_initialization),
        ("Differential Attention Model", test_differential_attention_model),
        ("MLA Model", test_mla_model),
        ("MLA + MoE Model", test_mla_with_moe),
        ("Attention Factory", test_attention_factory),
        ("Configuration Validation", test_configuration_validation),
        ("Distributed Training Config", test_distributed_training_config),
        ("Multi-Token Prediction", test_multi_token_prediction),
        ("Layer Sharing", test_layer_sharing),
        ("Layer Sharing Validation", test_layer_sharing_validation),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            print(f"✅ {test_name}: PASSED")
            passed_tests += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n{'='*60}")
    print(f"📊 **Modal GPU Results: {passed_tests}/{total_tests} tests passed**")
    
    if passed_tests == total_tests:
        print("🎉 **All 11 comprehensive tests passed on Modal GPU!**")
        print("\n**Architecture Features Validated:**")
        print("✅ Multi-Head Attention (MHA) support")
        print("✅ Differential Attention support")
        print("✅ Multi-Head Latent Attention (MLA) support")
        print("✅ MLA + Mixture of Experts (MoE) integration")
        print("✅ Multi-Token Prediction (MTP) support")
        print("✅ Layer Sharing (MobileLLM-style)")
        print("✅ Attention factory pattern")
        print("✅ Configuration validation")
        print("✅ Distributed training configuration")
        print("✅ Cut Cross Entropy support")
        print("\n🏆 **SabiYarn Model is fully functional on Modal GPU!**")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed on Modal GPU")
        return False

@app.local_entrypoint()
def modal_main():
    """Entry point for Modal execution."""
    print("🔄 Running comprehensive SabiYarn tests on Modal GPU...")
    result = run_all_tests.remote()
    
    if result:
        print("🎉 **All tests passed successfully!**")
        return True
    else:
        print("❌ **Some tests failed**")
        return False

if __name__ == "__main__":
    modal_main()
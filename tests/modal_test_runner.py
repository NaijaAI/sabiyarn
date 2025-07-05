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

# Add the project root to path so we can import sabiyarn as a package
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Create Modal app
app = modal.App("sabiyarn-tests")

@app.function(gpu="A10G", timeout=800)
def run_tests_on_gpu():
    """
    Run all tests on Modal GPU instance.
    This function runs on GPU, so all GPU-dependent imports are safe here.
    """
    import sys
    import os
    
    # Add the project root to path
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_root)
    
    # Now import the test functions (this happens on GPU)
    from tests.test_model_initialization import (
        test_cut_cross_entropy,
        test_mha_model_initialization,
        test_differential_attention_model,
        test_mla_model,
        test_mla_with_moe,
        test_attention_factory,
        test_configuration_validation,
    )
    
    print("🚀 **SabiYarn Model Initialization Tests (Modal GPU)**")
    print("=" * 60)
    
    test_functions = [
        ("Cut Cross Entropy", test_cut_cross_entropy),
        ("MHA Model Initialization", test_mha_model_initialization),
        ("Differential Attention Model", test_differential_attention_model),
        ("MLA Model", test_mla_model),
        ("MLA + MoE Model", test_mla_with_moe),
        ("Attention Factory", test_attention_factory),
        ("Configuration Validation", test_configuration_validation),
    ]
    
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
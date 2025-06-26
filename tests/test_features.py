import pytest
import torch
import modal
from sabiyarn.model import SabiYarn, ModelArgs, AttentionType
from sabiyarn.cut_cross_entropy import cut_cross_entropy_loss
from sabiyarn.MLA import MLA, MLAConfig

# Modal stub for running tests on Modal
stub = modal.Stub("sabiyarn-tests")

def run_on_modal(fn):
    """Decorator to run a test function on Modal GPU instance."""
    modal_func = stub.function(gpu="A10G", timeout=600)(fn)
    return modal_func.remote()

# --- MLA Test ---
def test_mla_forward():
    config = MLAConfig(
        hidden_size=512,
        num_heads=4,
        max_position_embeddings=128,
        rope_theta=10000,
        attention_dropout=0.1,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        kv_lora_rank=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        attention_bias=False,
    )
    mla = MLA(config)
    x = torch.randn(2, 32, 512)
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
    out, attn = mla(x, position_ids)
    assert out.shape == (2, 32, 512)
    assert attn.shape[0] == 2

# --- Cross Cut Entropy Test ---
def test_cut_cross_entropy():
    # needs to be worked on
    logits = torch.randn(4, 10, 100)
    targets = torch.randint(0, 100, (4, 10))
    loss = cut_cross_entropy_loss(logits, targets)
    assert loss.item() > 0

# --- Self-Attention Test ---
def test_self_attention_forward():
    args = ModelArgs(
        dim=256,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        max_seq_len=32,
        attention_type=AttentionType.SELF_ATTENTION,
    )
    model = SabiYarn(args)
    tokens = torch.randint(0, 100, (2, 16))
    hidden, logits = model(tokens, start_pos=0)
    assert logits.shape == (2, 16, 100)

# --- Modal Entrypoint for GitHub Actions ---
@stub.local_entrypoint()
def main():
    run_on_modal(test_mla_forward)
    run_on_modal(test_cut_cross_entropy)
    run_on_modal(test_self_attention_forward)
    print("All tests passed on Modal!")

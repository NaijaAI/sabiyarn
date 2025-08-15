#!/usr/bin/env python3
"""
Test generation using saved checkpoint from Modal training.
This helps debug dtype mismatches without interrupting training.

Usage:
  modal run test_generation.py::main
"""

import modal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Create Modal app with same image as training
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "transformers",
        "numpy",
        "structlog",
    ])
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app", ignore=[
        ".git", "*.pyc", "__pycache__", ".pytest_cache", "*.egg-info", 
        "out/", "*.bin", ".env"
    ])
)

app = modal.App("sabiyarn-generation-test")

# Same volume as training
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)

@app.function(
    gpu="A100-40GB",
    timeout=1200,
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_generation(
    prompts=None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    use_multi_token: bool = True,
    run_dir: str = None,
):
    """Generate text from the latest checkpoint on the Modal volume."""
    import os
    import sys
    from glob import glob

    os.chdir("/app")
    sys.path.insert(0, "/app")

    # Prefer persistent HF caches on the mounted volume
    cache_root = "/data/hf_cache"
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "transformers"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

    from sabiyarn.model import SabiYarn

    print("🔍 Running generation from checkpoint")

    # Locate checkpoint within per-run structure
    ckpt_base = "/data/checkpoints"
    pointer_file = os.path.join(ckpt_base, "LATEST_RUN.txt")
    resolved_run_dir = None
    if run_dir and os.path.isdir(run_dir):
        resolved_run_dir = run_dir
    elif os.path.exists(pointer_file):
        try:
            with open(pointer_file, "r") as fp:
                candidate = fp.read().strip()
                if candidate and os.path.isdir(candidate):
                    resolved_run_dir = candidate
        except Exception:
            resolved_run_dir = None
    if resolved_run_dir is None:
        # Fallback to most recent subdir under checkpoints
        subdirs = [d.path for d in os.scandir(ckpt_base) if d.is_dir()]
        if subdirs:
            resolved_run_dir = max(subdirs, key=lambda p: os.path.getmtime(p))
    ckpt_path = os.path.join(resolved_run_dir, "ckpt.pt") if resolved_run_dir else None
    if not ckpt_path or not os.path.exists(ckpt_path):
        # Final fallback: pick most recent *.pt anywhere under base
        all_pts = [p for p in glob(os.path.join(ckpt_base, "**", "*.pt"), recursive=True)]
        if not all_pts:
            print(f"❌ No checkpoints found in {ckpt_base}")
            return []
        ckpt_path = max(all_pts, key=lambda p: os.path.getmtime(p))
        print(f"⚠️ Using most recent checkpoint file by mtime: {ckpt_path}. Consider passing run_dir explicitly.")
    else:
        print(f"✅ Using run dir: {resolved_run_dir}")
        print(f"✅ Using checkpoint: {ckpt_path}")

    # Load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]

    # Normalize state_dict keys from compiled/DDP models
    raw_state = checkpoint["model"]
    needs_strip = any(k.startswith("_orig_mod.") or k.startswith("module.") for k in raw_state.keys())
    if needs_strip:
        normalized_state = {}
        for k, v in raw_state.items():
            new_k = k
            if new_k.startswith("_orig_mod."):
                new_k = new_k[len("_orig_mod."):]
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            normalized_state[new_k] = v
    else:
        normalized_state = raw_state

    model = SabiYarn(model_args)
    try:
        model.load_state_dict(normalized_state)
        print("✅ Model loaded successfully")
        print(f"Model: {model}")
    except Exception as e:
        print(f"⚠️ Strict load failed: {e}. Retrying with strict=False")
        missing, unexpected = model.load_state_dict(normalized_state, strict=False)
        print(f"   Missing keys: {missing}\n   Unexpected keys: {unexpected}")

    # Pick dtype by GPU capability; fall back to fp32 on CPU
    if device == "cuda":
        try:
            cc = torch.cuda.get_device_capability()
        except Exception:
            cc = (8, 0)
        preferred_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    else:
        preferred_dtype = torch.float32

    model = model.to(device, dtype=preferred_dtype)
    model.eval()

    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Aletheia-ng/SabiYarn-125M")

    # Defaults
    if prompts is None:
        prompts = [
            "<classify> Empeleni, amakhodi wendawo awasebenzi nakancane ekukopeni okungekho emthethweni; ukukopa kwe-bit nge-bit kwediski kuzoyenza idlale kahle kunoma iyiphi idivayisi lapho idiski yangempela ingadlala khona",
            "<prompt> Write a short poem about the ocean. <response>",
        ]

    outputs = []
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # Truncate to model context if needed
            if input_ids.size(1) > model_args.max_seq_len:
                input_ids = input_ids[:, -model_args.max_seq_len :]

            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_multi_token=use_multi_token and getattr(model, "use_multi_token", False),
            )

            text_full = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            text_input = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            completion = text_full[len(text_input) :]

            # print(f"\nPrompt {idx+1}: {prompt}")
            # print(f"Completion: {completion}")
            outputs.append({"prompt": prompt, "completion": completion, "full": text_full})

    return outputs

@app.local_entrypoint()
def main():
    """Run generation on Modal and print results."""
    print("🚀 Generating from latest checkpoint...")
    results = test_generation.remote()
    for r in results:
        print(f"\n=== Prompt ===\n{r['prompt']}\n=== Completion ===\n{r['completion']}")
    return True

if __name__ == "__main__":
    main()
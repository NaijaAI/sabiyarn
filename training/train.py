import os
import time
import math
from contextlib import nullcontext
import structlog
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F

from ..sabiyarn.model import ModelArgs, SabiYarn, MoeArgs
from ..sabiyarn.differential_attention import DiffAttnArgs
from ..cut_cross_entropy import linear_cross_entropy
from .utils import *
from .constant_tokens import MASK
from .training_attention_mask import create_causal_mask, create_causal_mask_ultra_optimized
import wandb
from torch.optim import SGD, Adam, AdamW
from bitsandbytes import optim  # For Adam8bit if using the bitsandbytes library
from omegaconf import OmegaConf

LOG = structlog.stdlib.get_logger()
# -----------------------------------------------------------------------------
# default config values

config = OmegaConf.load("../config/config.yaml")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb_key = os.getenv("WANDB_API_KEY")
hf_key = os.getenv("HF_WRITE_TOKEN")

# I/O
train_batch_size = config.training.train_batch_size
train_data_path = config.data.train_data_path
eval_data_path = config.data.eval_data_path
out_dir = config.training.out_dir
eval_interval = config.training.eval_interval
log_interval = config.training.log_interval
eval_iters = config.training.eval_iters
eval_only = config.training.eval_only  # if True, script exits right after the first eval
always_save_checkpoint = config.training.always_save_checkpoint  # if True, always save a checkpoint after each eval
init_from = config.model.init_from  # 'scratch' or 'resume'
use_cce = config.model.training.use_cce  # to use cut cross entropy or not

# wandb logging
wandb_log = config.wandb.log  # disabled by default
wandb_project = config.wandb.project
wandb_run_name = config.wandb.run_name  # 'run' + str(time.time())

# data
dataset = config.data.dataset
gradient_accumulation_steps = config.training.gradient_accumulation_steps # used to simulate larger batch sizes
batch_size = train_batch_size

# model
vocab_size = config.model.vocab_size
n_layers = config.model.n_layers
n_heads = config.model.n_heads
dropout = config.model.dropout  # for pretraining 0 is good, for finetuning try 0.model.1+
bias = config.model.bias  # do we use bias inside LayerNorm and Linear layers?
dim = config.model.dim
n_kv_heads = config.model.n_kv_heads
multiple_of = config.model.multiple_of # make SwiGLU hidden layer size multiple of large power of 2
ffn_dim_multiplier = config.model.ffn_dim_multiplier
norm_eps = config.model.norm_eps
use_moe = config.model.use_moe
moe = config.model.moe
attention_type = config.model.attention_type
logic_network = config.model.logic_network
max_batch_size = config.model.max_batch_size
max_seq_len = config.model.max_seq_len
block_size = config.model.block_size
use_j = config.model.use_j
display_model_output_iter = config.model.display_model_output_iter
num_experts = config.model.num_experts
num_experts_per_tok = config.model.num_experts_per_tok
tokenizer_name = config.model.tokenizer.name

# adamw optimizer
optimizer = config.optimizer.name
learning_rate = config.optimizer.learning_rate # max learning rate
max_iters = config.optimizer.max_iters  # total number of training iterations
weight_decay = config.optimizer.weight_decay
beta1 = config.optimizer.beta1
beta2 = config.optimizer.beta2
grad_clip = config.optimizer.grad_clip # clip gradients at this value, or disable if == config.0.0

# learning rate decay settings
decay_lr = config.training.decay_lr  # whether to decay the learning rate
warmup_iters = config.training.warmup_iters # how many steps to warm up for
lr_decay_iters = config.training.lr_decay_iters  # should be ~= config.max_iters per Chinchilla
min_lr = config.training.min_lr  # minimum learning rate, should be ~= config.learning_rate/10 per Chinchilla

# DDP settings
# os.environ["MASTER_ADDR"] = config."127.0.0.1"
# os.environ["MASTER_PORT"] = config."29500"
ddp = config.ddp.use_ddp
rank = config.ddp.rank
world_size = config.torch.cuda.device_count()
backend = config.ddp.backend


# system
device_type = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster


# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# will be useful for logging

# -----------------------------------------------------------------------------
# Don't add data download to this script because it keeps GPU idle!!!!!!!!!

# Start Training Model
LOG.info("starting training...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# various inits, derived attributes, I/O setup

if ddp:
    dist.init_process_group(
        backend=backend, rank=rank, init_method="env://", world_size=world_size
    )
    LOG.info("distributed processing initialized successfully....")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % world_size == 0
    gradient_accumulation_steps //= world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=wandb_project, config=config)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split, verbose=False):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    else:
        data = np.memmap(eval_data_path, dtype=np.uint16, mode="r")
        
    ix = torch.randint(len(data) - block_size, (train_batch_size,))
    x = [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    y = [
        torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
        for i in ix
    ]
    y = [
        process_labels_optimized(sample.clone(), MASK)
        # mask_long_sequences(process_labels(sample.clone(), MASK), mask_value=MASK)
        for sample in y
    ]
    x = torch.stack(x)
    y = torch.stack(y)

    if verbose:
        print("inputs: ", x)
        print("labels: ", y)

    # y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
        # x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        # x, y = x.to(device), y.to(device)
        x, y = x.to(device), y.to(device)

    return x, y


def configure_optimizer(
    model,
    optimizer_type="adam",
    learning_rate=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
):
    if optimizer_type == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "adam":
        optimizer = Adam(
            model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
        )

    elif optimizer_type == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
        )

    elif optimizer_type == "adam8bit":
        # Assuming you are using the "bitsandbytes" library for 8-bit optimization
        optimizer = optim.Adam8bit(
            model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
        )

    else:
        raise ValueError(
            f"Optimizer type '{optimizer_type}' not recognized. Please choose from 'sgd', 'adam', 'adamw', 'adam8bit'."
        )

    # Return the optimizer
    return optimizer


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

    # init a new model from scratch
print("Initializing a new model from scratch")
if use_moe:
    moe = MoeArgs(num_experts, num_experts_per_tok)

# Change the arguments to what is desirable.
if attention_type == "differential_attention":
    diff_attn_args = DiffAttnArgs(
        max_batch_size=max_batch_size,
        n_heads=n_heads,
        embed_dim=dim,
        n_kv_heads=n_kv_heads,
        max_seq_len=block_size,
        norm_eps=norm_eps,
    )

    model_args = ModelArgs(
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        moe,
        logic_network,
        max_batch_size,
        max_seq_len,
        use_j,
        attention_type,
        diff_attn_args,
    )
else:
    diff_attn_args = None
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
        norm_eps=norm_eps,
        moe=moe,
        logic_network=logic_network,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_j=use_j,
        attention_type=attention_type,
        diff_attn_args=diff_attn_args,
    )

model = SabiYarn(model_args)
LOG.info(f"{model.get_model_size()}")

# optimizer
optimizer = configure_optimizer(model, optimizer, weight_decay, betas=(beta1, beta2))

if init_from == "checkpoint":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
   
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    optimizer.load_state_dict(checkpoint["optimizer"])


model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == dtype))

checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
    
ddp_local_rank = rank
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


def _prepare_mask_(b=None, block_size=block_size, eval=False):
    if eval:
        return None

    attn_mask = torch.tril(torch.ones(block_size, block_size)).view(
        1, 1, block_size, block_size
    )

    attn_mask = attn_mask.repeat(b, 1, 1, 1)  # define attention_mask here.

    return attn_mask


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(use_cce: bool = False):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            b = len(X)
            with ctx:
                attn_mask = _prepare_mask_(b, block_size=block_size)
                attn_mask = create_causal_mask_ultra_optimized(X, attn_mask.to("cuda"))
                attn_mask.to(device)
                hidden_states, logits = model(tokens=X, mask=attn_mask, start_pos=0)
                if use_cce:
                    loss = linear_cross_entropy(
                        hidden_states,
                        model.lm_head.weight,
                        Y,
                        shift=True,
                        ignore_index = -100,
                        impl="torch_compile",
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100
                    )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def generate_and_decode_sequences(
    batch_token_ids, model, tokenizer, max_new_tokens=100
):
    # Ensure the model is in evaluation mode
    model.eval()

    # Generate new token sequences
    with torch.no_grad():
        generated_sequences = model.generate(
            idx=batch_token_ids,
            max_new_tokens=max_new_tokens,
        )

    # Decode the input and generated sequences
    decoded_inputs = tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)
    decoded_outputs = tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )

    # Print the input and corresponding generated sequence for each item in the batch
    for input_seq, output_seq in zip(decoded_inputs, decoded_outputs):
        print("Input Sequence: ")
        print(input_seq[-150:])
        print("-" * 80)
        print("Generated Sequence: ")
        print(output_seq.strip(input_seq))
        print("-" * 80)

    model.train()
    return


# training loop
def train():
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    global iter_num
    global best_val_loss

    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(use_cce)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }
                )
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        if iter_num % display_model_output_iter == 0 and master_process == 0:
            generate_and_decode_sequences(X, raw_model, tokenizer)

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                b = len(X)
                attn_mask = _prepare_mask_(b, block_size=config["block_size"])
                attn_mask = create_causal_mask(X, attn_mask)
                attn_mask.to("cuda")
                # print("Attention mask: ", attn_mask)
                hidden_states, logits = model(tokens=X, mask=attn_mask, start_pos=0)
                if use_cce:
                    loss = linear_cross_entropy(
                        hidden_states,
                        model.lm_head.weight,
                        Y,
                        ignore_index = MASK,
                        shift=True,
                        impl="torch_compile",
                    )
                else:

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=MASK
                    )
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            # if local_iter_num >= 5:  # let the training loop settle a bit
            # mfu = raw_model.estimate_mfu(
            #     batch_size * gradient_accumulation_steps, dt
            # )
            # running_mfu = (
            #     mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            # )
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms,")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()

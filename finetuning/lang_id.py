from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import structlog

LOG = structlog.stdlib.get_logger()

# data = load_dataset('Aletheia-ng/NLP-base-data',
#                     data_files=["language_identification.parquet"],
#                     split="train",
#                     token="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")

##finetuning for english

data = load_dataset(
    "damilojohn/lang_id_eng", token="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"
)
LOG.info("loaded dataset .....")

model = AutoModelForCausalLM.from_pretrained(
    "BeardedMonster/sabiyarn-125m", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("BeardedMonster/sabiyarn-125m")

action_token = tokenizer.encode("<lang_ID>")


device = "cuda" if torch.cuda.is_available() else "cpu"


def find_indices(tensor, numbers, pad_id):
    # Convert the list of numbers to a tensor for comparison
    numbers_tensor = torch.tensor(numbers)
    pad_id = torch.tensor(pad_id)

    # Create a mask for occurrences of the numbers in the tensor
    mask = torch.isin(tensor, numbers_tensor)
    pad_mask = torch.isin(tensor, pad_id)

    # Get the indices where the mask is True
    indices = torch.nonzero(mask, as_tuple=True)
    pad_indices = torch.nonzero(pad_mask, as_tuple=True)

    return indices[0], pad_indices[0]  # Return the first dimension indices


def preprocess_data(row):
    row["texts"] = [text.strip() for text in row["texts"]]
    out = {}

    tokens = tokenizer(
        row["texts"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
        add_special_tokens=True,
    )
    input_ids = tokens.input_ids
    pad = torch.full(
        (len(input_ids), 1), -100, dtype=input_ids.dtype, device=input_ids.device
    )
    labels = torch.cat([input_ids.clone()[:, 1:], pad], dim=1)
    for i in range(input_ids.size(0)):
        # Get first match of action token in this sample
        idx, pad_idx = find_indices(labels[i], action_token, pad_id=3)
        if idx.numel() != 0:
            labels[i, : idx[0] + 1] = -100

        if pad_idx.numel() != 0:
            labels[i, pad_idx[0] : pad_idx[-1]] = -100
    attention_masks = tokens.attention_mask

    out["targets"] = labels  # .to(torch.int8))
    out["idx"] = input_ids
    out["attn_mask"] = attention_masks

    out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}
    return out


tokenized_data = data.map(
    preprocess_data,
    batched=True,
    batch_size=64,
).with_format("torch")
LOG.info("...tokenizing dataset done")

split_data = tokenized_data["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_data["train"]
val_dataset = split_data["test"]


def create_causal_mask(padding_mask, device):
    """
    Create a causal attention mask from a padding mask.

    Args:
        padding_mask (torch.Tensor): The padding mask of shape (batch_size, seq_len),
                                    where 0 indicates padding and 1 indicates valid tokens.

    Returns:
        torch.Tensor: The causal mask of shape (batch_size, seq_len, seq_len),
                    where 0 allows attention and -inf prevents attention.
    """
    # Get the size of the batch and sequence length
    batch_size, seq_len = padding_mask.size()

    # Create a causal mask with ones on the lower triangle and zeros elsewhere
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))

    # Expand the causal mask to match the batch size
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    # Convert padding mask from (batch_size, seq_len) to (batch_size, 1, seq_len)
    # for broadcasting with causal mask
    padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1).to(device)

    # Combine causal mask with padding mask
    # If a token is padded (0 in padding_mask), set corresponding positions in causal_mask to -inf
    causal_mask = causal_mask.masked_fill(padding_mask == 0, 0)  # float('-inf'))

    return causal_mask.unsqueeze(1).to(bool)


batch_size = 16


def collate_fn(batch):
    # Convert each item in the batch to the appropriate tensor dtype
    inputs = torch.stack(
        [item["idx"].clone().detach().to(dtype=torch.int64) for item in batch]
    )
    labels = torch.stack(
        [item["targets"].clone().detach().to(dtype=torch.int64) for item in batch]
    )
    attn_mask = torch.stack(
        [item["attn_mask"].clone().detach().to(dtype=torch.int64) for item in batch]
    )
    return {"input_ids": inputs, "labels": labels, "attn_mask": attn_mask}


# Create a DataLoader with the custom collate function
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)


use_fp16 = False
gradient_accumulation_steps = 40
dtype = "bf16"

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.15)
num_train_steps = (
    len(train_dataloader) * 3 * gradient_accumulation_steps
)  # num_train_epochs = 3
scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=5e-6)
grad_clip = 1.0
# Training loop settings
num_epochs = 6
eval_steps = 16 * gradient_accumulation_steps
save_steps = 16 * gradient_accumulation_steps
train_losses = []
val_losses = []
best_val = 1e9
new_repo_name = "damilojohn/Sabiyarn_language_detection"

write_token = "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"


def train():
    LOG.info("Starting train.....")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        step_count = 0
        best_val = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = create_causal_mask(batch["attn_mask"], device).to(device)
            outputs = model(idx=inputs, targets=labels, attn_mask=attn_mask)
            # Store the unscaled loss for reporting
            unscaled_loss = outputs.loss.item()
            # Scale the loss for gradient accumulation
            loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            # Add the unscaled loss to the total
            total_loss += unscaled_loss
            step_count += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                # clip the gradient
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % eval_steps == 0 or (step == len(train_dataloader) - 1):
                LOG.info("Evaluating model.... ")
                model.eval()
                val_loss = 0
                val_step_count = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_inputs = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attn_mask = create_causal_mask(
                            val_batch["attn_mask"].to(device), device=device
                        )

                        val_outputs = model(
                            idx=val_inputs, targets=val_labels, attn_mask=val_attn_mask
                        )
                        val_loss += val_outputs.loss.item()
                        val_step_count += 1

                avg_val_loss = val_loss / val_step_count
                avg_train_loss = total_loss / step_count

                # Store losses for plotting
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

                LOG.info(
                    f"Epoch {epoch + 1}, Step {step + 1}/{len(train_dataloader)}: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}"
                )

                # Reset tracking for the next evaluation period
                if step != len(train_dataloader) - 1:  # Don't reset at end of epoch
                    total_loss = 0
                    step_count = 0

                model.train()
                if (step + 1) % save_steps == 0 and val_loss <= best_val:
                    best_val = val_loss
                    # Push to Hugging Face Hub
                    LOG.info("Pushing model to hub...")
                    model.push_to_hub(
                        new_repo_name,
                        commit_message=f"Saving model at step {step + 1}",
                        token=write_token,
                    )
    model.push_to_hub(
        new_repo_name, commit_message=f"final model save", token=write_token
    )

# # saves the openwebtext dataset to a binary file for training. following was helpful:
# # https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np

# import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files
import json
from .constant_tokens import end_of_text_token
import structlog
from dotenv import load_dotenv
load_dotenv()

load_dotenv()
LOG = structlog.stdlib.get_logger()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root =  os.path.dirname(current_dir)

config_path = os.path.join(project_root, "config", "config.yaml")
config = OmegaConf.load(config_path)

READ_TOKEN = os.getenv("HF_API_KEY")
num_proc = config.model.tokenizer.num_proc

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though

DATASETS = (
    
)

PROCESS_ONE_FILE_AT_A_TIME = True  #Should be True

def get_tokenizer_and_eot(tokenizer_name):
    """Initializes and returns the tokenizer and end_of_text_token."""
    enc = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eot_token = enc.eos_token_id if enc.eos_token_id is not None else end_of_text_token
    return enc, eot_token

def calculate_test_size(dataset_length):
    """Calculates the test split size based on dataset length."""
    if dataset_length <= 50: # Handle very small datasets, no test split
        return 0
    elif dataset_length <= 80000:
        return int(0.1 * dataset_length)
    elif dataset_length <= 800000:
        return int(0.01 * dataset_length)
    elif dataset_length <= 3500000:
        return int(0.0025 * dataset_length)
    else: # dataset_length > 3500000
        return int(0.0005 * dataset_length)

def process_example(example, tokenizer, eot_token):
    """Tokenizes a single example and adds end_of_text tokens."""
    ids = tokenizer.encode(example["text"])
    ids.extend([eot_token, eot_token])
    return {"ids": ids, "len": len(ids)}

def write_to_memmap(dset, filename, dtype, log_prefix=""):
    """Writes a dataset split to a memory-mapped file."""
    arr_len = np.sum(dset["len"], dtype=np.uint64)

    # Check if file exists to determine starting index for appending
    current_file_size_bytes = 0
    if os.path.exists(filename):
        current_file_size_bytes = os.path.getsize(filename)

    # Calculate initial idx based on existing file size in terms of elements
    initial_idx = current_file_size_bytes // np.dtype(dtype).itemsize

    # Check actual elements in the file to get `initial_idx` for 'r+'
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_arr = np.memmap(filename, dtype=dtype, mode="r")
        initial_idx = len(existing_arr)
        del existing_arr # Release file handle
        # Total size will be current size + new data size
        total_arr_len = initial_idx + arr_len
        arr = np.memmap(filename, dtype=dtype, mode="r+", shape=(total_arr_len,))
    else:
        initial_idx = 0
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    LOG.info(f"{log_prefix} created/opened bin file (current size: {initial_idx} elements, new data: {arr_len} elements)...")

    n_shards = min(1024, len(dset)) # Use min to avoid n_shards > len(dset) when dset is small

    current_idx = initial_idx
    LOG.info(f"{log_prefix} writing to bin file from index {initial_idx}...")

    for batch_idx in tqdm(range(n_shards), desc=f"{log_prefix} writing {filename}"):
        batch = dset.shard(
            num_shards=n_shards, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[current_idx : current_idx + len(arr_batch)] = arr_batch
        current_idx += len(arr_batch)
    arr.flush()
    LOG.info(f"{log_prefix} write to bin file complete...")


def run(datasets_list, num_proc_load_dataset):
    """
    Main function to process and tokenize datasets, saving to memory-mapped files.
    """
    files_processed = {}
    if os.path.exists("data_struct.json"):
        with open("data_struct.json", "r") as t:
            files_processed = json.load(t)
    else:
        LOG.info("The data_struct file does not exist. Starting fresh...")

    tokenizer, end_of_text_token = get_tokenizer_and_eot(config.model.tokenizer.name)

    # Define the process function once, outside the loop and with its dependencies
    # This avoids re-initializing tokenizer multiple times in multiprocessing
    def process_func_wrapper(example):
        return process_example(example, tokenizer, end_of_text_token)

    for dataset_name in datasets_list: # Use dataset_name to distinguish from loaded_dataset
        # This list will store files processed *for the current dataset_name* in this run
        current_dataset_processed_files = files_processed.get(dataset_name, [])

        if not PROCESS_ONE_FILE_AT_A_TIME:
            LOG.info(f"Downloading dataset '{dataset_name}'...")
            loaded_dataset = load_dataset(
                dataset_name,
                num_proc=num_proc_load_dataset,
                trust_remote_code=True,
                token=READ_TOKEN,
                verification_mode="no_checks",
            )
            # By default only contains the 'train' split, so create a test split
            train_split = loaded_dataset["train"]
            dataset_length = len(train_split)
            test_size = calculate_test_size(dataset_length)

            LOG.info("Creating splits...")
            if test_size == 0: # Very small dataset, no test split
                split_dataset = {"train": train_split, "val": train_split} # Use train for val as well
            else:
                split_dataset = train_split.train_test_split(
                    test_size=test_size, seed=2357, shuffle=True
                )
                split_dataset["val"] = split_dataset.pop("test") # Rename test to val

            LOG.info("Tokenizing splits...")
            tokenized_dataset = split_dataset.map(
                process_func_wrapper, # Use the wrapper here
                remove_columns=['text'],
                desc=f"tokenizing {dataset_name} splits",
                num_proc=num_proc,
            )
            LOG.info("Tokenizing and preprocessing complete..")

            LOG.info("Concatenating and binarizing splits...")
            for split, dset in tokenized_dataset.items():
                filename = config.train_data_path if split.lower() == "train" else config.eval_data_path
                write_to_memmap(dset, filename, np.uint16, log_prefix=f"[{dataset_name} - {split}]")
            
            # For this mode, consider the whole dataset as processed once done
            all_files_in_repo = list_repo_files(dataset_name, repo_type="dataset", token=READ_TOKEN)
            current_dataset_processed_files.extend(all_files_in_repo)
            # Remove duplicates if any
            current_dataset_processed_files = list(set(current_dataset_processed_files))


        else: # PROCESS_ONE_FILE_AT_A_TIME
            all_files = list_repo_files(dataset_name, repo_type="dataset", token=READ_TOKEN)
            
            files_to_process = [f for f in all_files if f not in current_dataset_processed_files]

            if not files_to_process:
                LOG.info(f"All files for dataset '{dataset_name}' already processed.")
                continue

            for file in files_to_process:
                LOG.info(f"Downloading and processing {file} from dataset '{dataset_name}'...")
                
                # IMPORTANT: Use dataset_name from the loop, not config["dataset"]
                loaded_dataset_file = load_dataset(
                    dataset_name,
                    data_files=[file],
                    num_proc=num_proc_load_dataset,
                    trust_remote_code=True, # Added this for consistency with first branch
                    token=READ_TOKEN, # Added this for consistency
                    verification_mode="no_checks",
                )

                train_split_file = loaded_dataset_file["train"]
                dataset_length = len(train_split_file)
                test_size = calculate_test_size(dataset_length)

                if dataset_length < 50 or test_size == 0:
                    split_dataset_file = {"train": train_split_file, "val": train_split_file} # Treat small datasets as both train/val
                    small_data = True
                else:
                    small_data = False
                    split_dataset_file = train_split_file.train_test_split(
                        test_size=test_size, seed=2357, shuffle=True
                    )
                    split_dataset_file["val"] = split_dataset_file.pop("test")

                LOG.info("Tokenizing splits...")
                tokenized_dataset_file = split_dataset_file.map(
                    process_func_wrapper, # Use the wrapper here
                    remove_columns=['text'], # This was commented out in original, but makes sense to keep
                    desc=f"tokenizing {file} splits",
                    num_proc=num_proc,
                )

                LOG.info("Concatenating and binarizing splits...")
                for split, dset in tokenized_dataset_file.items():
                    filename = config.train_data_path if split.lower() == "train" else config.eval_data_path
                    write_to_memmap(dset, filename, np.uint16, log_prefix=f"[{file} - {split}]")

                # Update processed files immediately after a file is successfully processed
                current_dataset_processed_files.append(file)
                files_processed[dataset_name] = current_dataset_processed_files # Update the main dict
                with open("data_struct.json", "w") as f:
                    json.dump(files_processed, f, indent=4) # Save progress
                LOG.info(f"Successfully processed and saved progress for {file}.")

        # After processing all files/the entire dataset, update the main files_processed dictionary
        # This is important for the `PROCESS_ONE_FILE_AT_A_TIME = False` case
        files_processed[dataset_name] = current_dataset_processed_files
        # For the `PROCESS_ONE_FILE_AT_A_TIME = False` case, we save after each dataset too
        # To avoid data loss if crash between datasets.
        if not PROCESS_ONE_FILE_AT_A_TIME:
             with open("data_struct.json", "w") as f:
                json.dump(files_processed, f, indent=4)
             LOG.info(f"Successfully processed and saved progress for entire dataset '{dataset_name}'.")

if __name__ == "__main__":
    run()
    LOG.info("data tokenization and processing complete.....")

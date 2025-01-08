# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
#import tiktoken
from datasets import load_dataset
import yaml
from transformers import AutoTokenizer
import json
from constant_tokens import end_of_text_token
import structlog
from huggingface_hub import login

LOG = structlog.stdlib.get_logger()
hf_key = os.getenv('HF_API_KEY')
login(token=hf_key)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
config = yaml.safe_load("sabiyarn_config.yaml")
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = config["num_proc"]

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc
enc = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True) #tiktoken.get_encoding("gpt2")


if __name__ == '__main__':
    if os.path.exists("data_struct.json"):
        with open("data_struct.json", "r") as t:
            files = json.load(t)
    else:
        print("The data_struct file does not exist. Loading files in the download order...")
        files = ""

    def process(example):
        enc = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True) 
        ids = enc.encode(example['text']) # encode_ordinary ignores any special tokens   encode_ordinary
        ids.append(end_of_text_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # print("Downloading and processing all files in the repository...")
    # dataset = load_dataset(config["dataset"], data_files=files, num_proc=num_proc_load_dataset,verification_mode='no_checks') #, split="train")

    # dataset_length = len(dataset["train"])
    # test_size = int(0.0005 * dataset_length)
    
    # # owt by default only contains the 'train' split, so create a test split
    # split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
    # split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # # tokenize the dataset
    # tokenized = split_dataset.map(
    #     process,
    #     remove_columns=['text'],
    #     desc="tokenizing the splits",
    #     num_proc=num_proc,
    #     fn_kwargs = {'enc':enc},
    # )

    if isinstance(files, str):
        print("Downloading and processing all files in the repository...")
        dataset = load_dataset(config["dataset"],
                               num_proc=num_proc_load_dataset,
                               verification_mode='no_checks')

        dataset_length = len(dataset["train"])
        test_size = int(0.01 * dataset_length)      
        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
            # fn_kwargs = {'enc':enc},
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            if split.lower() == 'train':
                filename = config["train_data_path"]  # os.path.join(os.path.dirname(__file__), f'{split}.bin')
            else:
                filename = config["eval_data_path"]
                
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            if os.path.exists(filename):
                arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(arr_len,))
            else:
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
                
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx:idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    else:
        # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        for file in files:
            print(f"Downloading and processing {file} ...")
            dataset = load_dataset(config["dataset"], data_files=[file], num_proc=num_proc_load_dataset,verification_mode='no_checks') #, split="train")

            dataset_length = len(dataset["train"])
            
            if dataset_length > 50 and dataset_length <= 80000:
                test_size = int(0.1 * dataset_length)
                
            elif dataset_length > 80000 and dataset_length <= 800000:
                test_size = int(0.01 * dataset_length)
            
            elif dataset_length > 800000 and dataset_length <= 3500000:
                test_size = int( 0.0025 * dataset_length)
                
            elif dataset_length > 3500000: 
                test_size = int(0.0005 * dataset_length)

            if dataset_length < 50:
                split_dataset = dataset
                small_data = True
            else:
                small_data = False
                # owt by default only contains the 'train' split, so create a test split
                split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
                split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

            # tokenize the dataset
            tokenized = split_dataset.map(
                process,
                remove_columns=['text'],
                desc="tokenizing the splits",
                num_proc=num_proc,
                # fn_kwargs={'enc':enc}
            )

            # concatenate all the ids in each dataset into one large file we can use for training
            for split, dset in tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                if split.lower() == 'train':
                    filename = config["train_data_path"]  # os.path.join(os.path.dirname(__file__), f'{split}.bin')
                else:
                    filename = config["eval_data_path"]
                    
                dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
                if os.path.exists(filename):
                    arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(arr_len,))
                else:
                    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

                if small_data:
                    total_batches = 1
                else:
                    total_batches = 32 if split == 'train' else 32

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                    # Batch together samples for faster write
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    # Write into mmap
                    arr[idx:idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()


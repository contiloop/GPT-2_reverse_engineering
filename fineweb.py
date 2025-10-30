"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".

NOTE: This script uses if __name__ == '__main__': to ensure safe multiprocessing,
which resolves the RuntimeError: An attempt has been made to start a new process...
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # Check for token range (optional but good practice)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def main():
    # 1. Setup
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # 2. Download the dataset
    print(f"Downloading dataset '{remote_name}'...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # 3. Setup multiprocessing pool
    # Use max(1, os.cpu_count()//2) to set process count for performance
    nprocs = max(1, os.cpu_count()//2)
    print(f"Starting tokenization with {nprocs} processes...")

    # 4. Tokenize documents and write shards
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        # pool.imap is used to distribute the 'tokenize' function across documents in 'fw'
        # chunksize=16 is used to manage the batching of documents sent to workers
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index:06d}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
                
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                progress_bar.close() # Close current progress bar
                
                shard_index += 1
                progress_bar = None
                
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # 5. write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])
            if progress_bar is not None:
                progress_bar.close()
    
    print(f"Tokenization complete. Total shards created: {shard_index + 1}")

if __name__ == '__main__':
    main()

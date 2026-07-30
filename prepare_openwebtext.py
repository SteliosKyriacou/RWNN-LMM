import os
import urllib.request
import numpy as np
import tiktoken

TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
VAL_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"

def download_file(url, filename):
    print(f"Downloading {filename} from GitHub raw...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        with open(filename, 'wb') as out_file:
            out_file.write(response.read())

def prepare_bpe_dataset():
    print("=== Preprocessing Text Dataset using GPT-2 BPE (tiktoken) ===")
    
    # 1. Download WikiText-2 raw files directly
    train_file = "wiki.train.raw"
    val_file = "wiki.valid.raw"
    
    if not os.path.exists(train_file):
        download_file(TRAIN_URL, train_file)
    if not os.path.exists(val_file):
        download_file(VAL_URL, val_file)
        
    # 2. Initialize tiktoken GPT-2 BPE Encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # 3. Process and Tokenize Train Split
    print("Tokenizing train split...")
    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    train_tokens = enc.encode_ordinary(train_text)
    print(f"Train set: {len(train_tokens):,} BPE tokens.")

    # 4. Process and Tokenize Val Split
    print("Tokenizing val split...")
    with open(val_file, "r", encoding="utf-8") as f:
        val_text = f.read()
    val_tokens = enc.encode_ordinary(val_text)
    print(f"Val set: {len(val_tokens):,} BPE tokens.")
    
    # 5. Save as binary files containing uint16 arrays (as in nanoGPT!)
    print("Saving to train.bin and val.bin...")
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    
    train_arr.tofile("train.bin")
    val_arr.tofile("val.bin")
    print("✓ Saved train.bin and val.bin successfully!")

if __name__ == "__main__":
    prepare_bpe_dataset()

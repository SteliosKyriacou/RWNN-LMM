import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_wikitext103():
    print("====================================================")
    print("          PREPARING WIKITEXT-103 DATASET            ")
    print("====================================================")
    
    # 1. Load Salesforce/wikitext via Hugging Face
    print("Downloading WikiText-103 raw from Salesforce/wikitext...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    # 2. Extract texts
    print("Extracting training and validation text splits...")
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])
    
    # 3. Initialize tiktoken GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    
    # 4. Tokenize Train Split
    print("Tokenizing training split (103 Million tokens - this might take 1-2 minutes)...")
    train_tokens = enc.encode_ordinary(train_text)
    print(f" -> Train set: {len(train_tokens):,} BPE tokens.")
    
    # 5. Tokenize Validation Split
    print("Tokenizing validation split...")
    val_tokens = enc.encode_ordinary(val_text)
    print(f" -> Val set: {len(val_tokens):,} BPE tokens.")
    
    # 6. Overwrite train.bin and val.bin
    print("Overwriting local train.bin and val.bin with WikiText-103...")
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    
    train_arr.tofile("train.bin")
    val_arr.tofile("val.bin")
    print("✓ Saved train.bin and val.bin successfully!")
    print("====================================================")

if __name__ == "__main__":
    prepare_wikitext103()
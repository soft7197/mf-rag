import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModel
import faiss
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)



def log_message(log_file, message):
    """ Append messages to a log file. """
    with open(log_file, "a") as log:
        log.write(message + "\n")

def split_into_chunks(text, tokenizer, max_tokens=512):
    """Dynamically split text into evenly distributed chunks if total tokens exceed max_tokens."""
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    if num_tokens <= max_tokens:
        return [tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)]

    # Calculate the number of chunks needed
    num_chunks = (num_tokens + max_tokens - 1) // max_tokens  # Equivalent to ceil(num_tokens / max_tokens)

    # Adjust chunk size to distribute tokens more evenly
    tokens_per_chunk = (num_tokens + num_chunks - 1) // num_chunks  # Equivalent to ceil division     

    chunks = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[i:i + tokens_per_chunk]), skip_special_tokens=True)
        for i in range(0, num_tokens, tokens_per_chunk)
    ]
    
    return chunks



def embed_text(text, model, tokenizer, device):
    """Convert text to a vector representation"""
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)  # Move tensors to device
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Ensure output is moved to CPU
    return embeddings


def process_bug(save_location, json_file, device, log_file):
    start_time = time.time()  # Track processing time
    bug_name = os.path.basename(json_file).split('_')[1].split('.')[0]

    try:
        # Load methods from JSON file
        with open(json_file, 'r') as f:
            method_data = json.load(f)

        if not method_data:
            log_message(log_file, f"[SKIPPED] {bug_name}: Empty file")
            return
        
        df = pd.DataFrame(method_data)
        methods = df['fullBody'].tolist()

        # Load CodeBERT
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

        method_chunks = []
        method_mapping = {} 

        method_id = 0
        
        embeddings = []
         
        for method_text in methods:
            chunks = split_into_chunks(method_text, tokenizer)
            for chunk in chunks:
                embedding = embed_text(chunk, model, tokenizer, device)
                method_mapping[len(method_chunks)] = method_id
                method_chunks.append(embedding)
                embeddings.append(embedding)
            method_id += 1

        embeddings = np.vstack(embeddings)

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Create directories
        os.makedirs(f"{save_location}/faiss_indexes", exist_ok=True)
        os.makedirs(f"{save_location}/mapping_metadata", exist_ok=True)
        os.makedirs(f"{save_location}/chunks_embeddings", exist_ok=True)

        # Save files
        base_name = f"bug_{bug_name}"
        faiss.write_index(index, f"{save_location}/faiss_indexes/{base_name}.index")
        with open(f"{save_location}/mapping_metadata/{base_name}_method_mapping.json", "w") as f:
            json.dump(method_mapping, f, indent=4)
        np.save(f"{save_location}/chunks_embeddings/{base_name}_method_chunks.npy", method_chunks)

        end_time = time.time()  # Track processing end time
        duration = round(end_time - start_time, 2)  # Processing time in seconds

        log_message(log_file, f"[SUCCESS] {bug_name}: Indexed {len(methods)} methods in {duration} seconds")
    
    except Exception as e:
        log_message(log_file, f"[FAILED] {bug_name}: {str(e)}")


if __name__ == "__main__":
    dir_path = ""
    save_location = ""
    log_file = f"{save_location}/processing_log.txt"

    # Get command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python embed_and_index.py <json_file> <gpu_id>")
        sys.exit(1)

    json_file = sys.argv[1]
    gpu_id = sys.argv[2]

    # Assign to specific GPU
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    process_bug(save_location, json_file, device, log_file)
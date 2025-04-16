import os
import faiss
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import traceback
import math

# Load CodeBERT Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)


# Paths
INDEX_DIR = ""
MAPPING_DIR = ""
METADATA_DIR = "s"
ERROR_LOG_FILE = ""


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

def query_similar_methods(query_code, method_mapping_json, index):
    # Step 1: Split the query code into chunks
    query_chunks = split_into_chunks(query_code, tokenizer)

    # Step 2: Compute embeddings for each chunk and normalize them
    query_embeddings = np.array([
        embed_text(chunk, model, tokenizer, device) for chunk in query_chunks
    ], dtype=np.float32)

    # Normalize query embeddings to ensure cosine similarity
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    methodid_chunkno_maxscore = {}  # Stores cosine similarity scores per method
    chunk_no = 0

    # Step 3: Retrieve nearest neighbors for each chunk
    for embedding in query_embeddings:
        embedding = embedding.reshape(1, -1)  # Reshape for cosine similarity computation

        # Retrieve all method embeddings from FAISS
        faiss_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

        # Normalize FAISS embeddings for cosine similarity
        faiss_embeddings /= np.linalg.norm(faiss_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity between query embedding and all method embeddings
        similarities = cosine_similarity(embedding, faiss_embeddings)[0]

        for idx in range (len(similarities)):
            method_id = method_mapping_json.get(str(idx), None)  # Convert idx to string if necessary
            if method_id is None:
                continue  # Skip missing method IDs
            
            if method_id not in methodid_chunkno_maxscore:
                methodid_chunkno_maxscore[method_id] = {}  # Initialize a new dictionary for chunks
            
            if chunk_no not in methodid_chunkno_maxscore[method_id]:
                methodid_chunkno_maxscore[method_id][chunk_no] = 0  # Initialize chunk similarity

            # Update max similarity score per method and embedding
            methodid_chunkno_maxscore[method_id][chunk_no] = max(
                methodid_chunkno_maxscore[method_id][chunk_no], similarities[idx]
            )
        
        chunk_no += 1  # Move to next chunk
    
    # Step 4: Compute final similarity scores (average over retrieved chunks)
    methodid_finalscore = {}
    for method_id in methodid_chunkno_maxscore:
        chunk_scores = list(methodid_chunkno_maxscore[method_id].values())
        methodid_finalscore[method_id] = sum(chunk_scores) / len(chunk_scores)  # Compute average similarity
    
    # Step 5: Retrieve top-K methods based on cosine similarity score
    top_method_ids = sorted(methodid_finalscore.items(), key=lambda x: x[1], reverse=True)
    
    itselfs_score = top_method_ids[0][1]
    
    index = 1 
    for i in range(len(top_method_ids)):
        if top_method_ids[i][1] != itselfs_score:
            index = i
            break

    return top_method_ids[index:]
    
def normalize_method_body(method_body):
    method_body = re.sub(r"/\*.*?\*/", " ", method_body, flags=re.DOTALL)
    method_body = re.sub(r"//.*", " ", method_body)
    method_body = re.sub(r"\r?\n", " ", method_body)
    method_body = re.sub(r"\s+", " ", method_body)
    method_body = method_body.replace("\"", "\\\"")
    return method_body.strip()

def extract_function_signature(code):
    # Updated regex to match Java method signatures (including generics, modifiers, annotations)
    function_pattern = re.compile(r"""
        ^\s*                                         # Optional leading spaces
        (?:@\w+\s*)*                                 # Optional annotations (e.g., @Override)
        (public|private|protected|static|final|abstract|synchronized|native|strictfp|\s)+  # Modifiers
        (\s*<[\w,\s\?]+>\s*)?                        # Optional generic type parameters (e.g., <T>)
        ([\w<>\[\],\?]+(?:\s+[\w<>\[\],\?]+)*)       # Return type (handles generics like Map<TypeVariable<?>, Type>)
        \s+(\w+)                                     # Method name
        \s*\([^)]*\)                                 # Parameter list
        (\s*throws\s+[\w<>\[\],.\s]+)?               # Optional throws clause
        """, re.VERBOSE | re.MULTILINE)

    # Updated regex to correctly detect constructors (including `final` parameters)
    constructor_pattern = re.compile(r"""
        ^\s*                                         # Optional leading spaces
        (public|private|protected)?\s*               # Constructor visibility modifiers (optional)
        ([A-Z][a-zA-Z0-9_]*)                         # Constructor name (class name)
        \s*\([^)]*\)                                 # Parameter list (allows `final` params)
        (\s*throws\s+[\w<>\[\],.\s]+)?               # Optional throws clause
        """, re.VERBOSE | re.MULTILINE)

    # Normalize code to remove line breaks for better matching
    cleaned_code = " ".join(code.splitlines()).strip()

    # First check for constructors (since they don't have return types)
    match = constructor_pattern.search(cleaned_code)
    if match:
        return match.group(0).strip(), False

    # If no constructor is found, check for a method signature
    match = function_pattern.search(cleaned_code)
    return (match.group(0).strip(), True) if match else (None, None)


def normalize_function_signature2(signature):
    # Remove extra spaces around parentheses and between words
    signature = re.sub(r"\s*\(\s*", "(", signature)  # Remove spaces before/after '('
    signature = re.sub(r"\s*,\s*", ", ", signature)  # Ensure single space after commas
    signature = re.sub(r"\s*\)\s*", ")", signature)  # Remove spaces before/after ')'
    signature = re.sub(r"\s+", " ", signature)       # Collapse multiple spaces into one

    return signature.strip()

def compute_similarity(str1, str2):
    """ Compute similarity score between two strings using SequenceMatcher. """
    return SequenceMatcher(None, str1, str2).ratio()
bug_function = {}
def extract_normalized_buggy_function_body(bug, bug_json, data2):
    if "functions" in bug_json:
        no=0
        for func in bug_json["functions"]:
            buggy_func = func["buggy_function"]
            extracted_signature, isfunction = extract_function_signature(buggy_func)
            if extracted_signature is None:
                if bug not in bug_function:
                    bug_function[bug]=[]
                bug_function[bug].append(no)
                print(f"{bug}--{no}------------------------------------------------------------------------------------")
                continue
            if not isfunction:
                func["isConstructor"] = True
                func["normalized_body"] = func["buggy_function"]
                continue
            extracted_signature = normalize_method_body(extracted_signature)
            extracted_signature = normalize_function_signature2(extracted_signature)
            func["normalized_body"] = []
            for method in data2:
                str2 = normalize_method_body(method["fullSignature"])
                str2 = normalize_function_signature2(str2)
                a = str2 == extracted_signature
                b=func["path"] in method["filePath"]
                if a and b:
                    func["normalized_body"].append(method["fullBody"])
            no+=1
    else:       
        buggy_func = bug_json["buggy"]
        extracted_signature, isfunction = extract_function_signature(buggy_func)
        if extracted_signature is None:
            bug_function[bug]= " "
            print("--------------------------------------------------------------------------------------")
            return 
        if not isfunction:
                bug_json["isConstructor"] = True
                bug_json["normalized_body"] = bug_json["buggy"]
                return
        extracted_signature = normalize_method_body(extracted_signature) 
        bug_json["normalized_body"] = []
        for method in data2:
            if method["fullSignature"] == extracted_signature and bug_json["loc"] in method["filePath"]:
                bug_json["normalized_body"].append(method["fullBody"])
import random

def allocate_similar_methods(num_buggy_methods: int, total_to_retrieve: int) -> list:
    if total_to_retrieve < num_buggy_methods:
        raise ValueError("Total methods to retrieve must be at least equal to number of buggy methods")

    base = total_to_retrieve // num_buggy_methods
    remainder = total_to_retrieve % num_buggy_methods

    # Start with base allocation for all
    allocation = [base] * num_buggy_methods

    # Distribute the remainder one by one
    for i in range(remainder):
        allocation[i] += 1

    return allocation


def retrieve_context(bug_id, query_methods):
    try:
        print(f"{bug_id} ========= retirieving start")
        with open("/home/selab/Desktop/MF-dlt/resources/defects4j_full.json", "r") as file:
            bugs = json.load(file)

        with open(f"{MAPPING_DIR}/bug_{bug_id}_method_mapping.json", "r") as file:
            mapping_json = json.load(file)

        with open(f"{METADATA_DIR}/methods_{bug_id}.json", "r") as file:
            metadata_json = json.load(file)
        index = faiss.read_index(f"{INDEX_DIR}/bug_{bug_id}.index")
        
        retrieved_methods = []

        if "functions" in bugs[bug_id]:
            function_num = len(query_methods)

            if function_num > 5:
                for func in query_methods:
                    query_method = func["normalized_body"][0]
                    top_methods_id_scores = query_similar_methods(query_method, mapping_json, index)[:1]                   
                    for id_score in top_methods_id_scores:
                        retrieved_methods.append([id_score[1], metadata_json[id_score[0]]['fullBody']])
            else:
                allocation = allocate_similar_methods(function_num, 5)
                ind = 0
                for func in query_methods:
                    query_method = func["normalized_body"][0]
                    top_methods_id_scores = query_similar_methods(query_method, mapping_json, index)[:allocation[ind]]                   
                    for id_score in top_methods_id_scores:
                        retrieved_methods.append([id_score[1], metadata_json[id_score[0]]['fullBody']])
                    ind+=1

        else:
            retrieve_num = 5
            query_method = bugs[bug_id]["normalized_body"][0]
            top_methods_id_scores = query_similar_methods(query_method, mapping_json, index)[:retrieve_num]           
            for id_score in top_methods_id_scores:
                retrieved_methods.append([id_score[1], metadata_json[id_score[0]]['fullBody']])

        print(f"Retrieval is done")

    except Exception as e:
        # Save the error message to a log file
        with open(ERROR_LOG_FILE, "a") as error_file:
            error_file.write(f"Error in bug {bug_id}:\n{traceback.format_exc()}\n{'-'*80}\n")

        print(f"Error in {bug_id}, logged to {ERROR_LOG_FILE}. Continuing...")
    
    return retrieved_methods


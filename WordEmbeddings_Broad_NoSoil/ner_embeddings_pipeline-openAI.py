
import os
import json
import argparse
from typing import List, Dict, Any
import sys
from transformers import AutoConfig
sys.path.append(os.path.abspath('..'))


import torch
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from Evaluation_Files.generate_ner_prompt_nosoil_definition import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple_excel_partial_exact_count  import evaluate_all
from dotenv import load_dotenv
import os

import time
import random

load_dotenv()

def load_dataset(path: str):
    data = []
    texts = []
    entities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            input_field = obj.get("input", "")
            text_val = input_field.split("\n### Text ###\n", 1)[-1] if "\n### Text ###\n" in input_field else input_field
            output_field = obj.get("output", "")
            try:
                ents = json.loads(output_field) if isinstance(output_field, str) else output_field
            except Exception:
                ents = []
            data.append({"text": text_val, "entities": ents})
            texts.append(text_val)
            entities.append(ents)
    return data, texts, entities

def build_or_load_index(embeddings: np.ndarray, index_path: str):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    if index_path:
        faiss.write_index(index, index_path)
    return index

def embed_texts(embedder, texts: List[str], batch_size: int = 64):
    embs = embedder.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return embs

def get_similar_examples(query_text: str, embedder, index, corpus_texts: List[str], corpus_entities: List[Any], top_k: int = 5):
    q = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, top_k)
    idxs = I[0].tolist()
    return [(corpus_texts[i], corpus_entities[i]) for i in idxs], (D[0].tolist(), idxs)



def perform_ner(text, max_length, examples, max_retries=5, base_delay=1):
    system_prompt, user_prompt = generate_ner_prompts(text)
    blocks = []
    for i, (example_text, ents) in enumerate(examples):
        blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
    blocks_str = "\n\n".join(blocks)
    system_prompt += blocks_str
    
    api_key = os.getenv("OPENAI_API_KEY")
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "gpt-5",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    "max_completion_tokens": max_length,
                    "reasoning_effort": "minimal",
                })
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response. Raw response: {response.text}")
                    raise
            
            # Handle rate limiting (status code 429)
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    # Extract retry-after header if available
                    retry_after = response.headers.get('retry-after')
                    if retry_after:
                        delay = int(retry_after)
                        print(f"Rate limited. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
                    else:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited. Waiting {delay:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    
                    time.sleep(delay)
                    continue
                else:
                    print(f"Max retries ({max_retries}) reached for rate limiting")
                    response.raise_for_status()
            
            # Handle other HTTP errors
            else:
                print(f"API Error - Status Code: {response.status_code}")
                print(f"Response Text: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Request failed: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Max retries ({max_retries}) reached for request errors")
                raise
    
    raise Exception(f"Failed to complete request after {max_retries} attempts")


def process_text_files(input_dir, output_dir, max_length, embedder, index, corpus_texts, corpus_entities, top_k):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            if os.path.exists(output_text_path):
                with open(output_text_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    data = json.loads(text)  
                    content = (
                        data.get('choices', [{}])[0]       # Get first choice or default to [{}]
                            .get('message', {})            # Get 'message' dict or empty dict
                            .get('content')                # Get 'content' or None
                    )

                    if content and content.strip():
                        print(f"Skipping {filename} (already processed).")
                        continue

            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            examples, _ = get_similar_examples(text, embedder, index, corpus_texts, corpus_entities, top_k)


            print(f"Processing {filename}...")
            ner_result = perform_ner(text, max_length, examples)

            with open(output_text_path, "w", encoding="utf-8") as file:
                json.dump(ner_result, file, ensure_ascii=False, indent=2)

            print(f"Saved: {output_text_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset with 'input' and 'output'")
    ap.add_argument("--index", default="ner_embeddings.index", help="FAISS index output path")
    ap.add_argument("--embedder", default="intfloat/multilingual-e5-large", help="Embedding model")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--input_dir", required=True, help="Directory containing input text files.")
    ap.add_argument("--output_dir", required=True, help="Directory to save annotated output files.")
    ap.add_argument("--output_dir_json", required=True, help="Directory to save extracted json output from annotated files.")
    ap.add_argument("--model_name", required=True, help="Name of the model used for annotation.")
    ap.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")
    args = ap.parse_args()

    # Load data
    data, corpus_texts, corpus_entities = load_dataset(args.dataset)

    # Build embeddings
    embedder = SentenceTransformer(args.embedder)
    corpus_embs = embed_texts(embedder, corpus_texts)

    # Build index
    index = build_or_load_index(corpus_embs, args.index)
    
    
    process_text_files(args.input_dir, args.output_dir, args.max_length, embedder, index, corpus_texts, corpus_entities, top_k=args.top_k)
    log_dir = os.environ.get('LOG_DIR')

    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptemberDocumentLevel/Test_XMI_Files"
    evaluate_all(args.model_name,
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        int(args.top_k),
        xmi_dir,
        log_dir)
    print("NER processing complete.")
    
    

if __name__ == "__main__":
    main()

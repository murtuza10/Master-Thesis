
import os
import json
import argparse
from typing import List, Dict, Any
import sys
from transformers import AutoConfig

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt_broad_definition import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple_excel_partial_exact  import evaluate_all



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



def perform_ner(model, tokenizer, text, max_length, examples):
    system_prompt, user_prompt = generate_ner_prompts(text)
    blocks = []
    for i, (example_text, ents) in enumerate(examples):
        blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
    blocks_str = "\n\n".join(blocks)
    system_prompt += blocks_str

    # Build messages list according to role-based chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    print("messages:", messages)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize the prompt
    input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1500,
        temperature=0.7,
        top_p=0.9,)

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


def process_text_files(input_dir, model, tokenizer, output_dir, max_length, embedder, index, corpus_texts, corpus_entities, top_k):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            if os.path.exists(output_text_path):
                print(f"Skipping {filename} (already processed).")
                continue

            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            examples, _ = get_similar_examples(text, embedder, index, corpus_texts, corpus_entities, top_k)


            print(f"Processing {filename}...")
            ner_result = perform_ner(model, tokenizer, text, max_length, examples)

            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)

            print(f"Saved: {output_text_path}")

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Then load model with clean config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Successfully loaded in FP16")
    return model, tokenizer


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
    ap.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    ap.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")
    args = ap.parse_args()

    # Load data
    data, corpus_texts, corpus_entities = load_dataset(args.dataset)

    # Build embeddings
    embedder = SentenceTransformer(args.embedder)
    corpus_embs = embed_texts(embedder, corpus_texts)

    # Build index
    index = build_or_load_index(corpus_embs, args.index)
    
    
    model, tokenizer = load_model(args.model_path)
    process_text_files(args.input_dir, model, tokenizer, args.output_dir, args.max_length, embedder, index, corpus_texts, corpus_entities, top_k=args.top_k)
    
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

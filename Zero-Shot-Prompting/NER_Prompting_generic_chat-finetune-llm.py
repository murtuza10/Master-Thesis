import os
from pyexpat.errors import messages
import sys
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.quantization_config import QuantizationMethod
from transformers import AutoConfig
import bitsandbytes as bnb
from peft import PeftModel

sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple import evaluate_all



def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
#     bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # Or "fp4"
#     bnb_4bit_compute_dtype="float16",  # Use float16 or bfloat16 depending on hardware
# )

    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Will place model on the correct GPU/CPU automatically
).eval()
    
#     # 2. Add new tokens (if any were added during fine-tuning)
    new_tokens = [ "</tool_call>", "<tool_call>", "<|box_end|>", "<|box_start|>", "<|endoftext|>", "<|file_sep|>" ,
  "<|fim_middle|>" ,
  "<|fim_pad|>",
  "<|fim_prefix|>",
  "<|fim_suffix|>",
  "<|im_end|>",
  "<|im_start|>",
  "<|image_pad|>",
  "<|object_ref_end|>",
  "<|object_ref_start|>",
  "<|quad_end|>",
  "<|quad_start|>",
  "<|repo_name|>",
  "<|video_pad|>",
  "<|vision_end|>",
  "<|vision_pad|>",
  "<|vision_start|>" ]  
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))  # Expands embedding layer

    adapter_path = "/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/autotrain-Qwen2-5-14B-Document-7-Aug"
    model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer

def perform_ner(model, tokenizer, prompt_text, max_length):
    """
    Performs NER inference using the given model and tokenizer.

    Args:
        model: Huggingface transformer model.
        tokenizer: Tokenizer for the model.
        prompt_text (str): Full prompt text to send to the model (e.g., instruction + input).
        max_length (int): Maximum input token length.

    Returns:
        str: The model's decoded response.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages = [
    {"role": "user", "content": prompt_text}
]
    # Tokenize the prompt
    input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(input_ids=input_ids,
    attention_mask=attention_mask,max_new_tokens=1024)



    # Decode and return
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    # Optional: postprocess to extract just the JSON part
    # json_start = response.find('{')
    # json_end = response.rfind('}') + 1
    # if json_start != -1 and json_end != -1:
    #     response = response[json_start:json_end]

    return response


def process_jsonl_file(jsonl_path, model, tokenizer, output_prediction_path, output_gold_path, max_length):
    os.makedirs(os.path.dirname(output_prediction_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_gold_path), exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_prediction_path, "w", encoding="utf-8") as f_pred, \
         open(output_gold_path, "w", encoding="utf-8") as f_gold:

        for i, line in enumerate(f_in, 1):
            item = json.loads(line)
            input_text = item["input"]
            gold_output = item["output"]

            print(f"Processing line {i}...")

            # Call the model
            prediction = perform_ner(model, tokenizer, input_text, max_length)

            # Save outputs
            f_pred.write(json.dumps({"id": i, "prediction": prediction}, ensure_ascii=False) + "\n")
            f_gold.write(json.dumps({"id": i, "gold_output": gold_output}, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Perform NER using a transformer-based model.")
    parser.add_argument("--jsonl_path", required=True, help="Path to JSONL file with input/output examples.")
    parser.add_argument("--output_pred", required=True, help="Path to save model predictions.")
    parser.add_argument("--output_gold", required=True, help="Path to save actual gold output.")
    parser.add_argument("--model_name", required=True, help="Name of the model used for annotation.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    parser.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")

    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)

    process_jsonl_file(
        jsonl_path=args.jsonl_path,
        model=model,
        tokenizer=tokenizer,
        output_prediction_path=args.output_pred,
        output_gold_path=args.output_gold,
        max_length=args.max_length
    )



if __name__ == "__main__":
    main()

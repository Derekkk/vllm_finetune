import json
import re
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig
import requests
from flask import Flask, render_template, request, jsonify
from instruction_generation import *
from peft import PeftModel


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = 'cuda'

llava_7b_path = "llava-hf/llava-v1.6-mistral-7b-hf"
# llava_7b_path = "llava-hf/llava-v1.6-vicuna-13b-hf"
# llava_7b_path = "llava-hf/llava-v1.6-34b-hf"

print(llava_7b_path)

llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    llava_7b_path, 
    torch_dtype=torch.float16, 
    # load_in_4bit=True, 
    device_map="auto",
    # cache_dir="/data/huzhe/workspace/model_card/llavanext34b"
    )
llava_processor = LlavaNextProcessor.from_pretrained(
    llava_7b_path, 
    # cache_dir="/data/huzhe/workspace/model_card/llavanext34b"
    )

# lora_weights = "../save_model/sft-human-centered-v2-llava-v1.6-mistral-7b-hf/checkpoint-4000"
# llava_model = PeftModel.from_pretrained(
#     llava_model,
#     lora_weights,
#     torch_dtype=torch.bfloat16,
# )
llava_model.eval()
print(llava_model)


def llava_inference(instruction, image_path):
    # # 7b
    # prompt = f"[INST] <image>\n{instruction}[/INST]"
    # 13b
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{instruction} ASSISTANT:"

    image = Image.open(image_path)
    image = None
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)
    # Generate
    generate_ids = llava_model.generate(
        **inputs, 
        max_new_tokens=1024,
        )
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path

    task = "caption" # feedback, mcq_withnorm, trajectory, mcq_withtrajectory, mcq_oracle_norm, norm_entailment
    read_path = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/data_annotation_v2.json"
    image_folder = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/images_v2_all/"

    write_path = f"../results/result_llava-v1.6-mistral-7b-hf_{task}.json"
    print("write_path: ", write_path)

    data = json.load(open(read_path))
    results = []

    
    for sample in data:
        # instructions = sample["instructions"] 
        instructions = formulate_instruction(sample, None, task)

        image_path = image_folder + sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            print("[sample]: ", instruction)
            pred = llava_inference(instruction, image_path)
            if "[INST]" in pred and "[/INST]" in pred:
                pred = pred.split("[/INST]")[1].strip()
            pattern1 = r'USER:(.*?)\nASSISTANT:' 
            pattern2 = r'USER:(.*?)ASSISTANT:'
            pred = re.sub(pattern1, '', pred).strip()
            pred = re.sub(pattern2, '', pred).strip()
            if "USER" in pred and "ASSISTANT:" in pred:
                pred = pred.split("ASSISTANT:")[1].strip()
            # if "ASSISTANT: " in pred:
            #     pred = pred.split("ASSISTANT: ")[1].strip()
            cur_preds.append({"instruction": instruction, "prediction": pred})
            print("[pred]: ", pred)
        sample["result"] = cur_preds
        sample["caption"] = cur_preds[0]["prediction"]
        results.append(sample)
    
    print("save_to: ", write_path)
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    # data = json.load(open("data/multimodal_questions.json"))
    # for sample in data:
    #     print("[sample]: ", sample)
    #     instruction = sample["instructions"] 
    #     image_path = "data/" + sample["image_file"]
    #     result = llava_inference(instruction, image_path)
    #     print("[result]: ", result)
    #     break

    main()
    

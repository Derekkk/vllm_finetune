import json
import re
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor

from PIL import Image
import requests
import copy
import torch
import tqdm
import sys
import warnings
from instruction_generation import *
from peft import PeftModel


device = 'cuda'

warnings.filterwarnings("ignore")
model_id = "/data1/huzhe/workspace/model_cards/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# lora_weights = "../save_model/sft-human-centered-v2-llava-onevision-qwen2-7b-ov-hf/checkpoint-4000"
# model = PeftModel.from_pretrained(
#     model,
#     lora_weights,
#     torch_dtype=torch.bfloat16,
# )

model.eval()
print(model)


def model_inference(instruction, image_path):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            # {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # raw_image = Image.open(image_path)
    raw_image = None
    inputs = processor(images=raw_image, text=prompt, add_special_tokens=False, return_tensors='pt').to(device)

    output = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)

    text_outputs = processor.batch_decode(output, skip_special_tokens=True)

    return text_outputs[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path

    task = "mcq_textonly" # feedback, mcq_withnorm, trajectory, mcq_withtrajectory, mcq_oracle_norm, norm_entailment
    # read_path = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/data_annotation_v2.json"
    read_path = "../results/result_Llama-3.2-11B-Vision-Instruct_caption.json"
    image_folder = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/images_v2_all/"

    write_path = f"../results/result_Llama-3.2-11B-Vision-Instruct_{task}_oracle.json"

    print("write_path: ", write_path)

    data = json.load(open(read_path))
    results = []

    
    for sample in tqdm.tqdm(data):

        instructions = formulate_instruction(sample, None, task)

        image_path = image_folder + sample["image_file"]
        print(sample["image_file"])
        cur_preds = []
        for instruction in instructions:
            print("[sample]:\n", instruction)
            pred = model_inference(instruction, image_path)
            if "[INST]" in pred and "[/INST]" in pred:
                pred = pred.split("[/INST]")[1].strip()
            pattern1 = r'USER:(.*?)\nASSISTANT:' 
            pattern2 = r'USER:(.*?)ASSISTANT:'
            pred = re.sub(pattern1, '', pred).strip()
            pred = re.sub(pattern2, '', pred).strip()
            if "USER" in pred and "ASSISTANT:" in pred:
                pred = pred.split("ASSISTANT:")[1].strip()
            if "user" in pred and "assistant" in pred:
                pred = pred.split("assistant")[1].strip()
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

    main()
    

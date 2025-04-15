import json
import re
from PIL import Image
import torch
import requests
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from instruction_generation import *
from peft import PeftModel
import re
from loguru import logger
import tqdm


device = 'cuda'

warnings.filterwarnings("ignore")
model_id = "/data/huzhe/workspace/model_card/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

lora_weights = "../save_model/textual_reason/gpt4v110k_llava-onevision-qwen2-7b-ov-hf/checkpoint-900"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.bfloat16,
)
model.eval()
print(model)


def llava_inference(instruction, image_path):
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
    logger.debug(f"[input]: {[prompt]}")
    raw_image = Image.open(image_path)
    raw_image = None
    logger.debug(f"[raw_image]: {[raw_image]}")
   
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

    output = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)

    text_outputs = processor.batch_decode(output, skip_special_tokens=True)
    logger.debug(f"[output]: {[text_outputs]}")
    return text_outputs[0]


def extract_reason_and_answer(text):
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip().strip("*")
    return {"reason": reason, "answer": answer}


def formulate_instruction_mcq(sample_dict):
    option_str = "\n".join(sample_dict["action_list"])
    cur_input = f'''You are given a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \nShown in the given image.\n\n## Question:\nSelect the most appropriate course of initial action to take\n{option_str}\n\nNow answer the question. Just output the choice:'
'''
    return cur_input.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path

    task = "mcq" # feedback, mcq_withnorm, trajectory, mcq_withtrajectory, mcq_oracle_norm, norm_entailment
    read_path = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/data_annotation_v2.json"
    image_folder = "/data/huzhe/workspace/multimodal_llm/data_empathy/benchmark/v2/images_v2_all/"

    # write_path = f"../results/result_llava-onevision-qwen2-7b-ov_llama3.1_30kdata_{task}_4090_run2.json"
    write_path = f"../results/results_reason/result_gpt4v210k_llava-onevision-qwen2-7b-ov-{task}.json"

    logger.debug(f"write_path: {write_path}")

    data = json.load(open(read_path))
    results = []

    
    for sample in tqdm.tqdm(data):
        # instructions = formulate_instruction(sample, None, task)
        instructions = [formulate_instruction_mcq(sample)]

        image_path = image_folder + sample["image_file"]
        # image_path = sample["image_file"]
        cur_preds = []
        for instruction in instructions:
            pred = llava_inference(instruction, image_path)
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

            # parse reasoning model
            if "</think>" in pred and "<think>" in pred:
                pred = extract_reason_and_answer(pred)
                logger.debug(f"- parsed output:\n{pred}\n")
                cur_preds.append({"instruction": instruction, "prediction": pred})
            else:
                logger.debug(f"- output:\n{pred}\n")
                cur_preds.append({"instruction": instruction, "prediction": pred})

        sample["result"] = cur_preds
        results.append(sample)
    
    logger.debug(f"save_to: {write_path}")

    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":

    # prompt = "What is the color of the car?"
    # result = llava_inference(prompt, None)
    # print(result)

    main()
    

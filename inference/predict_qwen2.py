import json
import re
from PIL import Image
import torch
import requests
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor

from qwen_vl_utils import process_vision_info

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from instruction_generation import *
from peft import PeftModel
from loguru import logger
import tqdm


device = 'cuda'

# model_path = "Qwen/Qwen2-VL-7B-Instruct"
model_path = "/data3/huzhe/workspace/model_cards/Qwen2.5-VL-3B-Instruct"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

lora_weights = "../save_model/textual_reason/gpt4v110k_Qwen2.5-VL-3B-Instruct/checkpoint-300"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.bfloat16,
)

model.eval()
print(model)


# default processer
processor = AutoProcessor.from_pretrained(model_path)


def qwen2_inference(instruction, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logger.debug(f"[input]: {[text]}")
    image_inputs, video_inputs = process_vision_info(messages)
    # image_inputs, video_inputs = None, None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



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

    image_folder = "/data/huzhe/workspace/multimodal_llm/data_empathy//benchmark/v2/images_v2_all/"

    write_path = f"../results/results_reason/result_gpt4v210k_Qwen2.5-VL-3B-Instruct-{task}.json"

    print("write_path: ", write_path)

    data = json.load(open(read_path))
    results = []

    
    for sample in tqdm.tqdm(data):
        # instructions = formulate_instruction(sample, None, task)
        instructions = [formulate_instruction_mcq(sample)]
        image_path = image_folder + sample["image_file"]

        cur_preds = []
        for instruction in instructions:

            pred = qwen2_inference(instruction, image_path)
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
                logger.debug(f"- parsed output:\n{[pred]}\n")
                cur_preds.append({"instruction": instruction, "prediction": pred})
            else:
                logger.debug(f"- output:\n{[pred]}\n")
                cur_preds.append({"instruction": instruction, "prediction": pred})

        sample["result"] = cur_preds
        results.append(sample)
    
    logger.debug(f"save_to: {write_path}")
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":

    main()
    

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

import torch
import os
import functools
from datasets import load_dataset
import logging
import json
import copy
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
from peft import PeftModel
import re
import tqdm
from loguru import logger
from vllm import LLM, SamplingParams


os.environ["VLLM_ATTENTION_BACKEND"] = "triton"

# model_path = "/data3/huzhe/workspace/model_cards/Qwen2.5-1.5B-Instruct/"
model_path = "../save_model/llms/GPT4o_v2_10kdata_Qwen2.5-1.5B-Instruct_sft/checkpoint-162"

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def extract_reason_and_answer(text):
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip().strip("*")
    return {"reason": reason, "answer": answer}


def generate_one(situation):
   
    messages = [
        # {"role": "system", "content": "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."},
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": situation,}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.debug(f"- Input:\n{[text]}")

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    generated_text = outputs[0].outputs[0].text

    return generated_text


def formulate_instruction_mcq_text(sample_dict):
    option_str = "\n".join(sample_dict["action_list"])
    cur_input = f'''You are given a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \n{sample_dict["caption"]}\n\n## Question:\nSelect the most appropriate course of initial action to take\n{option_str}\n\nNow answer the question. Just output the choice:'
'''
    return cur_input.strip()
    

def viva_action_selection(read_path, write_path):
    data = json.load(open(read_path))
    data_pred = []

    for sample in tqdm.tqdm(data):
        cur_input = formulate_instruction_mcq_text(sample)
        # logger.debug(f"- prompt:\n{[cur_input]}")
        output = generate_one(cur_input.strip())
        logger.debug(f"- original output:\n{[output]}\n")
        if "</think>" in output and "<think>" in output:
            output = extract_reason_and_answer(output)
            logger.debug(f"- parsed output:\n{output}\n")
            sample["model_output"] = output["answer"]
            sample["reason"] = output["reason"]
        else:
            logger.debug(f"- output:\n{output}\n")
            sample["model_output"] = output
        cur_preds = [{"instruction": cur_input, "prediction": output}]
        sample["result"] = cur_preds
        data_pred.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(data_pred, f_w, indent=2)


def gpt_data_action_selection(read_path, write_path):
    print(f"read_path: {read_path}")
    print(f"write_path: {write_path}")
    data = [json.loads(ln) for ln in open(read_path)][40000:]
    data_pred = []

    for sample in tqdm.tqdm(data):
        cur_input = sample["messages"][0]["content"][0]["text"]
        logger.debug(f"- prompt:\n{[cur_input]}")
        output = generate_one(cur_input.strip())
        logger.debug(f"- original output:\n{[output]}\n")
        if "</think>" in output and "<think>" in output:
            output = extract_reason_and_answer(output)
            logger.debug(f"- parsed output:\n{output}\n")
            sample["model_output"] = output["answer"]
            sample["reason"] = output["reason"]
        else:
            logger.debug(f"- output:\n{output}\n")
            sample["model_output"] = output
        cur_preds = [{"instruction": cur_input, "prediction": output}]
        sample["result"] = cur_preds
        data_pred.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(data_pred, f_w, indent=2)


if __name__ == "__main__":
    viva_benchmark_path = "/data3/huzhe/workspace/human_centered_decisionmaking/vlms/results/results_textual/result_llava-onevision-qwen2-7b-ov_finetunedv2_caption.json"
    write_path = "../results/llm/viva_llm_GPT4o_v2_10kdata_Qwen2.5-1.5B-Instruct_sft.json"
    viva_action_selection(viva_benchmark_path, write_path)

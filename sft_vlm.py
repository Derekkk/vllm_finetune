# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, LlavaOnevisionForConditionalGeneration

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import logging
import sys
from qwen_vl_utils import process_vision_info
import json
import re

# from configs import (
#     DataArguments,
#     H4ArgumentParser,
#     ModelArguments,
#     SFTConfig,
#     get_peft_config
# )


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    logger.setLevel(logging.INFO) 

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        # use_cache=False if training_args.gradient_checkpointing else True,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def clean_assistant_response(example):
        # Parse the input JSON
        # Loop through the messages to find the assistant's response
        for item in example["messages"]:
            if item['role'] == 'assistant':
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        # Remove <think>...</think> and keep only <answer>...</answer>
                        text = content_item['text']
                        
                        # Extract the answer content
                        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                        
                        if answer_match:
                            # Replace the entire text with just the answer content
                            content_item['text'] = answer_match.group(1).strip()
        return example

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False).replace("gievn", "given").replace("<\\s>", processor.tokenizer.eos_token).strip() for example in examples]
        # print([texts[0]])
        images = [example["images"] for example in examples]
        # if isinstance(model, LlavaForConditionalGeneration):
        #     # LLava1.5 does not support multiple images
        #     images = [image[0] for image in images]
        images = None
        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        # image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch


    def collate_fn_qwen2(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False).replace("gievn", "given") for example in examples]

        # texts = [processor.apply_chat_template(clean_assistant_response(example["messages"]), tokenize=False).replace("gievn", "given") for example in examples]

        images = [example["images"] for example in examples]

        images = None
        video_inputs = None
        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, videos=video_inputs, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        # image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="/data1/huzhe/workspace/hf_cache")

    train_dataset = load_dataset(
        "json", 
        data_files="./data/textual_decision_making/reason_data/Reason_Distill_gpt4o_situation_mcq_v2_train_trainable_processed.jsonl", 
        cache_dir="/data1/huzhe/workspace/hf_cache",
        split="train[:10000]"
        )
    test_dataset = load_dataset(
        "json", 
        data_files="./data/textual_decision_making/reason_data/Reason_Distill_gpt4o_situation_mcq_v2_dev_trainable_processed.jsonl",
        cache_dir="/data1/huzhe/workspace/hf_cache",
        split="train"
    )
    train_dataset = train_dataset.remove_columns("reason").remove_columns("result").remove_columns("old_messages")
    test_dataset = test_dataset.remove_columns("reason").remove_columns("result").remove_columns("old_messages")
    
    train_dataset = train_dataset.map(clean_assistant_response)
    test_dataset = test_dataset.map(clean_assistant_response)
    
    # train_dataset = load_dataset(
    #     "json", 
    #     data_files="./data/textual_decision_making/Llama-3.1-8B-Instruct_situation_mcq_all_extracted_train_trainable.jsonl", 
    #     cache_dir="/data1/huzhe/workspace/hf_cache",
    #     split="train[:30000]"
    # )
    # test_dataset = load_dataset(
    #     "json", 
    #     data_files="./data/textual_decision_making/Llama-3.1-8B-Instruct_situation_mcq_all_extracted_dev_trainable.jsonl",
    #     cache_dir="/data1/huzhe/workspace/hf_cache",
    #     split="train"
    # )

    dataset = DatasetDict({"train":train_dataset, "test": test_dataset})
    print(dataset)


    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in dataset.items()]}"
    )
    logger.info(
        f"Training sample:\n {dataset[script_args.dataset_train_split][0]}"
    )

    peft_config = get_peft_config(model_args)
    # peft_config.target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    print(peft_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn_qwen2,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
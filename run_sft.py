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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )


    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="/data1/huzhe/workspace/hf_cache")

    def process_datset(example, tokenizer):
        #  {'messages': [{'role': 'user', 'content': [{'text': 'You are gievn a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \nYou are a high school teacher supervising a group project competition where three teams are presenting. One team accuses another of copying their project idea during the presentations. The accused team denies it, but there are similarities in their work. A decision must be made whether to pause the competition to investigate the claim, allow the presentations to continue and investigate after, or dismiss the accusation as unfounded given the lack of clear evidence. Pausing the competition could frustrate all students, but dismissing the claim could be perceived as unfair.\n\n## Question:\nHow should you handle the situation?\nA. Pause the competition immediately to thoroughly investigate the claim.\nB. Allow the competition to continue but promise to investigate the claim afterward.\nC. Dismiss the accusation, citing the lack of evident proof from the accuser.\nD. Hold a group discussion among all teams to address the accusation before continuing.\n\nNow answer the question. Just output the choice:', 'type': 'text'}]}, {'role': 'assistant', 'content': [{'text': 'B', 'type': 'text'}]}], 'images': [None]}

        message = []
        for msg in example["messages"]:
            cur_msg_proc = {"role": msg["role"], "content": msg["content"][0]["text"]}
            message.append(cur_msg_proc)
        chat_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        example_proc = {"text": chat_text}
        return example_proc


    train_dataset = load_dataset(
        "json", 
        data_files="./data/textual_decision_making/GPT4o_situation_mcq_v2_train_trainable.jsonl", 
        cache_dir="/data1/huzhe/workspace/hf_cache",
        split="train[:10000]"
        )
    test_dataset = load_dataset(
        "json", 
        data_files="./data/textual_decision_making/GPT4o_situation_mcq_v2_dev_trainable.jsonl",
        cache_dir="/data1/huzhe/workspace/hf_cache",
        split="train"
    )

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

    column_names = list(train_dataset.features)
    train_dataset = train_dataset.map(
        process_datset,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        remove_columns=column_names,
        desc="Applying chat template",
    )
    test_dataset = test_dataset.map(
        process_datset,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        remove_columns=column_names,
        desc="Applying chat template",
    )
    
    print(f"train_dataset: {train_dataset[0]}")
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
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
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

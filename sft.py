# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# Full training
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub

# LoRA
CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch sft.py \
    --model_name_or_path /data/huzhe/workspace/model_card/Llama-3.1-8B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT
"""

import argparse

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


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    def process_datset(example, tokenizer):
        message = []
        for msg in example["messages"]:
            cur_msg_proc = {"role": msg["role"], "content": msg["content"][0]["text"].replace("gievn", "given")}
            message.append(cur_msg_proc)
        example_proc = {"messages": message}
        return example_proc


    # train_path = "./data/textual_decision_making/GPT4o_situation_mcq_v2_train_trainable.jsonl"
    # validation_path = "./data/textual_decision_making/GPT4o_situation_mcq_v2_dev_trainable.jsonl"

    train_path = "../../data/textual_decision_making/reason_data/Reason_Distill_gpt4o_situation_mcq_v2_train_trainable_processed.jsonl"
    validation_path = "../../data/textual_decision_making/reason_data/Reason_Distill_gpt4o_situation_mcq_v2_dev_trainable_processed.jsonl"

    train_dataset = load_dataset(
        "json", 
        data_files=train_path, 
        split="train[:10000]"
        )
    test_dataset = load_dataset(
        "json", 
        data_files=validation_path,
        split="train"
    )


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
    print(dataset["train"][0])
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
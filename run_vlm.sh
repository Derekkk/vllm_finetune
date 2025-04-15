# Tested on 8x H100 GPUs
# /data/huzhe/workspace/model_card/llava-onevision-qwen2-7b-ov-hf
# llava-hf/llava-v1.6-mistral-7b-hf
# /data1/huzhe/workspace/model_cards/Llama-3.2-11B-Vision-Instruct
CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch  --config_file=./accelerate_configs/deepspeed_zero2.yaml \
    sft_vlm.py  \
    --dataset_name ./data/llava-instruct-mix-vsft \
    --model_name_or_path /data3/huzhe/workspace/model_cards/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir ./save_model/textual_reason/gpt4v110k_noreason_Qwen2.5-VL-3B-Instruct \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --num_train_epochs 3 \
    --eval_steps 300 \
    --save_steps 300 \
    --max_length 8000 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_target_modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

# CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --config_file=./accelerate_configs/multi_gpu.yaml sft_vlm.py ./train_configs/llavanext_config_qlora.yaml --load_in_4bit=true

# For LLaVA-NeXT, use: (requires transformers>=4.45)
#     --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

# For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
#     --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct


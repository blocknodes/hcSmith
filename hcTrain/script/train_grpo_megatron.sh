#!/usr/bin/env bash
# hcAgent 拆单 GRPO · ms-swift Megatron。单轮 Env(plan_env.py)，reward 只按 domain。
# 数据: train_grpo.jsonl (每行 {"messages":[...], "env_config":{"name":"plan_grpo","query":...,"expected_domains":[...]}})
# 用法: DRY_RUN=1 bash train_grpo_megatron.sh   |   bash train_grpo_megatron.sh
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
MODEL="${MODEL:-/opt/ai/input/data/juice/prd_ai_center/train_project/dingbo4/models/Qwen3.5-35B-A3B}"
DATASET="${DATASET:-${SCRIPT_DIR}/train_grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/grpo_hcagent_plan}"
GPUS="${GPUS:-1,3}"
EP="${EP:-2}"
MAX_LEN="${MAX_LEN:-6144}"
GLOBAL_BS="${GLOBAL_BS:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"

CMD=(
    megatron rlhf
    --rlhf_type grpo
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
    --external_plugins "${SCRIPT_DIR}/plan_env.py"
    --use_gym_env true
    --gym_env plan_grpo
    --multi_turn_scheduler gym_scheduler
    --max_turns 1
    --enable_thinking false
    --tuner_type lora
    --lora_rank 8
    --lora_alpha 32
    --target_modules all-linear
    --context_parallel_size 1
    --tensor_model_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size "${EP}"
    --sequence_parallel false
    --moe_permute_fusion true
    --moe_grouped_gemm true
    --moe_shared_expert_overlap true
    --moe_aux_loss_coeff 1e-4
    --num_train_epochs "${EPOCHS}"
    --global_batch_size "${GLOBAL_BS}"
    --micro_batch_size 1
    --max_length "${MAX_LEN}"
    --lr "${LR}"
    --lr_warmup_fraction 0.05
    --min_lr 1e-6
    --padding_free true
    --bf16 true
    --attention_backend flash
    --recompute_granularity full
    --recompute_method uniform
    --recompute_num_layers 1
    --freeze_vit true
    --freeze_aligner true
    --freeze_llm false
    --finetune true
    --save_strategy epoch
    --save_total_limit 10
    --save_safetensors true
    --no_save_optim false
    --no_save_rng false
    --load_from_cache_file true
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --logging_steps 1
    --report_to tensorboard
)

NPROC=$(awk -F',' '{print NF}' <<< "${GPUS}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=%q CUDA_VISIBLE_DEVICES=%q ' "${NPROC}" "${GPUS}"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE="${NPROC}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
"${CMD[@]}"
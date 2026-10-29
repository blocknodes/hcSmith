#!/usr/bin/env bash
# hcAgent 拆单 DPO · ms-swift Megatron。参数与 serv/train/dpo.sh 验证的一致。
# MODEL=... GPUS=1,3 EPOCHS=3 LR=1e-5 bash train_dpo_megatron.sh
# DRY_RUN=1 只打印命令不执行。
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
MODEL="${MODEL:-/opt/ai/input/data/juice/prd_ai_center/train_project/dingbo4/models/Qwen3.6-35B-A3B}"
# 训练数据：环境变量 DATASET 显式指定（单文件），不再自动合并数据源。
DATASET="${DATASET:?需要通过 DATASET=... 指定训练数据 jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/dpo_hcagent_plan}"
GPUS="${GPUS:-1,3}"
# MoE 专家并行度跟随进程/卡数（EP 必须能整除卡数，否则 Megatron 校验报错）
NPROC=$(awk -F',' '{print NF}' <<< "${GPUS}")
EP="${EP:-${NPROC}}"
MAX_LEN="${MAX_LEN:-6144}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-6}"
MIN_LR="${MIN_LR:-1e-6}"
BETA="${BETA:-0.3}"

CMD=(
    megatron rlhf
    --rlhf_type dpo
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
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
    --micro_batch_size "${MICRO_BS}"
    --max_length "${MAX_LEN}"
    --lr "${LR}"
    --min_lr "${MIN_LR}"
    --lr_warmup_fraction 0.05
    --beta "${BETA}"
    --rpo_alpha 1.0
    --split_dataset_ratio 0.05
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
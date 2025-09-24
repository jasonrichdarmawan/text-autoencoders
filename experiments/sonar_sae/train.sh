#!/bin/bash

set -a
source $WORKSPACE/text-autoencoders/.env
set +a

CMD="python $(dirname "$0")/train.py \
    --mode $MODE"

CMD="$CMD --d_sae $D_SAE \
    --total_training_batches 30_000 \
    --lr $LR \
    --lr_warm_up_steps 3_000 \
    --lr_decay_steps 6_000 \
    --batch_size 128 \
    --accumulate_grad_batches 32 \
    \
    --device $CUDA_ID \
    \
    --logger_dir $WORKSPACE/experiments/sonar_sae \
    --logger_name $LOGGER_NAME \
    \
    --checkpoints_dir $WORKSPACE/experiments/sonar_sae/checkpoints"

if [ "$SAE_TYPE" == "gated" ]; then
    CMD="$CMD --sae_type gated \
        --l1_coefficient $L1_COEFFICIENT \
        --l1_warm_up_steps 3_000"
elif [ "$SAE_TYPE" == "batchtopk" ]; then
    CMD="$CMD --sae_type batchtopk \
        --k $K"
elif [ "$SAE_TYPE" == "jump_relu" ]; then
    CMD="$CMD --sae_type jump_relu \
        --l0_coefficient $L0_COEFFICIENT"
fi

if [ "$MODE" == "load_from_checkpoint" ]; then
    CMD="$CMD --checkpoint_filename $CHECKPOINT_FILENAME"
fi

eval $CMD
#!/bin/bash

set -a
source $WORKSPACE/text-autoencoders/.env
set +a

CMD="python $(dirname "$0")/train.py \
    --mode $MODE"

CMD="$CMD --d_sae $D_SAE \
    --total_training_batches $TOTAL_TRAINING_BATCHES \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    \
    --device $CUDA_ID \
    \
    --logger_dir $WORKSPACE/experiments/sonar_sae \
    --logger_name $LOGGER_NAME \
    \
    --checkpoints_dir $WORKSPACE/experiments/sonar_sae/checkpoints"

if [ "$SAE_TYPE" == "gated" ]; then
    CMD="$CMD --sae_type gated \
        --lr_warm_up_steps $LR_WARM_UP_STEPS \
        --lr_decay_steps $LR_DECAY_STEPS \
        --l1_coefficient $L1_COEFFICIENT \
        --l1_warm_up_steps $L1_WARM_UP_STEPS"
elif [ "$SAE_TYPE" == "batchtopk" ]; then
    CMD="$CMD --sae_type batchtopk \
        --k $K"
elif [ "$SAE_TYPE" == "jump_relu" ]; then
    CMD="$CMD --sae_type jump_relu \
        --lr_decay_steps $LR_DECAY_STEPS \
        --l0_coefficient $L0_COEFFICIENT \
        --l0_warm_up_steps $L0_WARM_UP_STEPS"
fi

if [ "$MODE" == "load_from_checkpoint" ]; then
    CMD="$CMD --checkpoint_filename $CHECKPOINT_FILENAME"
fi

eval $CMD
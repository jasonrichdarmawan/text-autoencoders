#!/bin/bash

CMD="python train.py \
    --workspace $WORKSPACE \
    --mode $MODE"

if [ "$MODE" == "load_from_dict" ]; then
    CMD="$CMD --d_sae 16384 \
        --l1_coefficient $L1_COEFFICIENT \
        --l1_warm_up_steps 3_000 \
        --total_training_batches 30_000 \
        --lr $LR \
        --lr_warm_up_steps 3_000 \
        --lr_decay_steps 6_000 \
        --batch_size 128 \
        --accumulate_grad_batches 32 \
        \
        --devices $CUDA_ID \
        \
        --logger_dir $WORKSPACE/experiments/sonar_sae \
        --logger_name $LOGGER_NAME \
        \
        --checkpoints_dir $WORKSPACE/experiments/sonar_sae/checkpoints"
fi

eval $CMD
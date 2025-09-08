#!/bin/bash

CMD="python autointerp.py \
    --mode $MODE \
    --result_filename $WORKSPACE/experiments/sonar_sae/autointerp_results/$LOGGER_ID/$CHECKPOINT_NAME-nllb-200-6M-sample-embedding.json"

if [[ -n "$NUM_SHARDS" && -n "$SHARD_IDX" ]]; then
    # SHARD_SIZE=$((D_SAE / NUM_SHARDS))
    # START=$((SHARD_IDX * SHARD_SIZE))
    # END=$((START + SHARD_SIZE - 1))
    # LATENTS_LIST=$(seq $START $END)
    LATENTS_LIST=$(seq $SHARD_IDX $NUM_SHARDS $((D_SAE - 1)))
elif [[ $LATENTS == *-* ]]; then
    # Check if step is provided (format: start-end-step)
    if [[ $(echo $LATENTS | tr '-' '\n' | wc -l) -eq 3 ]]; then
        START=$(echo $LATENTS | cut -d'-' -f1)
        END=$(echo $LATENTS | cut -d'-' -f2)
        STEP=$(echo $LATENTS | cut -d'-' -f3)
        END_EXCLUSIVE=$((END - 1))
        LATENTS_LIST=$(seq $START $STEP $END_EXCLUSIVE)
    else
        # Default step of 1 (format: start-end)
        START=$(echo $LATENTS | cut -d'-' -f1)
        END=$(echo $LATENTS | cut -d'-' -f2)
        END_EXCLUSIVE=$((END - 1))
        LATENTS_LIST=$(seq $START $END_EXCLUSIVE)
    fi
else
    LATENTS_LIST=$LATENTS
fi

if [ "$MODE" == "autointerp" ]; then
    CMD="$CMD --d_sae $D_SAE \
        --batch_size 4096 \
        --latents $LATENTS_LIST \
        --checkpoint_filename $WORKSPACE/experiments/sonar_sae/checkpoints/$LOGGER_ID/$CHECKPOINT_NAME.ckpt \
        --device cuda:2 \
        --max_concurrent 128"
fi

eval $CMD
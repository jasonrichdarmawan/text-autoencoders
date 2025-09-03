#!/bin/bash

CMD="
python autointerp.py \
    --mode $MODE \
    --result_filename $WORKSPACE/experiments/sonar_sae/autointerp_results/$LOGGER_NAME/$CHECKPOINT_NAME-nllb-200-6M-sample-embedding.json \
"

if [[ $LATENTS == *-* ]]; then
    START=$(echo $LATENTS | cut -d'-' -f1)
    END=$(echo $LATENTS | cut -d'-' -f2)
    LATENTS_LIST=$(seq $START $END)
else
    LATENTS_LIST=$LATENTS
fi

if [ "$MODE" == "autointerp" ]; then
    CMD="
    $CMD --d_sae 16384 \
         --batch_size 128 \
         --latents $LATENTS_LIST \
         --checkpoint_filename $WORKSPACE/experiments/sonar_sae/checkpoints/$LOGGER_NAME/$CHECKPOINT_NAME.ckpt \
         --device cuda:2 \
         --max_concurrent 11 \
    "
fi

eval $CMD
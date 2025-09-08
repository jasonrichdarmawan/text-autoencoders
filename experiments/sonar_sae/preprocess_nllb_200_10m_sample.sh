#!/bin/bash

CMD="python preprocess_nllb_200_10m_sample.py \
    --workspace $WORKSPACE \
    --split $SPLIT \
    --num_shards $NUM_SHARDS"

if [ -n "$SAVE" ]; then
    CMD="$CMD --mode save_to_disk \
        --shard_idx $SHARD_IDX \
        --cudaId $CUDA_ID"
fi

if [ -n "$PUBLISH" ]; then
    SHARD_PATHS=""
    for ((i=0; i<NUM_SHARDS;i++)); do
        SHARD_PATHS="$SHARD_PATHS $WORKSPACE/data/nllb-200-6M-sample-embedding/$SPLIT-$(printf '%05d' $i)-of-$(printf '%05d' $NUM_SHARDS)"
    done
    CMD="$CMD --mode push_to_hub \
        --shard_paths $SHARD_PATHS"
fi

eval $CMD
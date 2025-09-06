#!/bin/bash

CMD="
python preprocess_nllb_primary_datasets.py \
    --mode $MODE \
"

if [ "$MODE" == "save_to_disk" ]; then
    CMD="
    $CMD --data_dir $DATA_DIR \
         --cudaId $CUDA_ID \
         --dataset_name $DATASET_NAME \
         --data_loader_batch_size 128 \
         --predict_batch_size 8 \
         --num_shards $NUM_SHARDS \
         --shard_idx $SHARD_IDX \
         --save_dir $SAVE_DIR \
    "
fi

if [ "$MODE" == "push_to_hub" ]; then
    CMD="
    $CMD --processed_dir $PROCESSED_DIR \
    "
fi

eval $CMD
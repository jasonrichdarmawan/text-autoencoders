#!/bin/bash

CMD="
python main.py \
  --data_dir $DATA_DIR \
  --default_root_dir $DEFAULT_ROOT_DIR \
  --accelerator gpu \
  --strategy ddp \
  --devices 2 3 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_warmup_steps 1237 \
  --max_steps 123742 \
"

if [ -n "$CHECKPOINT_PATH" ]; then
  CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
fi

eval $CMD
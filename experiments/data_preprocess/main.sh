python main.py \
--data_dir $DATA_DIR \
--accelerator gpu \
--strategy ddp \
--devices 1 2 3 \
--batch_size 8 \
--lr 1e-4 \
--num_warmup_steps 824 \
--max_steps 82494 \
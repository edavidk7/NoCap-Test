# Context-length curriculum, as "frac:len,frac:len,..." over num_iterations.
# Empty is the baseline: a constant --sequence_length.
# Curriculum variant: SEQ_LEN_SCHEDULE="0:256,0.3:512,0.6:1024"
SEQ_LEN_SCHEDULE="0:64,0.05:128,0.15:512,0.25:1024"

uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --seq_len_schedule $SEQ_LEN_SCHEDULE \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --log_wandb

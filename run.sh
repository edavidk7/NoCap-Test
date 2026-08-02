GRAD_ACCUM_SCHEDULE="0:2,0.05:4,0.1:8,0.2:16,0.35:32"

uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 16 \
  --grad_accum_schedule "$GRAD_ACCUM_SCHEDULE" \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 6000 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --log_wandb \
  --ff_kind glu \
  --seed 42 \
  --mlp_drop_n 2 \
  --mlp_alpha 6.0 \
  --mlp_act silu \
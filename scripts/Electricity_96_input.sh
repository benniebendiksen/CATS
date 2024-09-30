export CUDA_VISIBLE_DEVICES=0

model_name=CATS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.2 \
  --lradj TST \
  --pct_start 0.4 \
  --batch_size 32

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.3 \
  --lradj TST \
  --pct_start 0.4 \
  --batch_size 32

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.5 \
  --lradj TST \
  --pct_start 0.4 \
  --batch_size 32

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.7 \
  --lradj TST \
  --pct_start 0.4 \
  --batch_size 32
#!/bin/bash

# Set up Pyenv with the desired Python version
export PATH="$HOME/.pyenv/versions/3.10.16/bin:$PATH"
export PYENV_VERSION=3.10.16

# Debugging: Ensure the right Python version is being used
echo "============================================================"
echo "Checking Python version..."
echo "Expected Python version: 3.10.16"
current_python_version=$(python --version)
echo "Current Python version: $current_python_version"

# Safety check: Ensure the correct Python version is active
if [[ "$current_python_version" != *"3.10.16"* ]]; then
  echo "Error: Expected Python version 3.10.16, but found $current_python_version."
  exit 1
fi

# Check if the correct version of PyTorch is available
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('Torch CUDA Available:', torch.cuda.is_available())"

model_name=CATS

# First model configuration
data_path="btcusdt_pca_components_12h_60_07_05.csv"
seq_len=96
pred_len=1
dec_in=63
d_model=128
patch_len=24
stride=24
data_file=$(basename "$data_path" .csv)

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/logits/ \
  --data_path $data_path \
  --model_id "1_${data_file}_${seq_len}_${pred_len}_${dec_in}" \
  --model $model_name \
  --data logits \
  --features MS \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --d_layers 3 \
  --dec_in $dec_in \
  --des 'Logits' \
  --d_model $d_model \
  --d_ff 256 \
  --patch_len $patch_len \
  --stride $stride \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 5 \
  --train_epochs 50 \
  --patience 7 \
  --exp_name logits \
  --target close \
  --is_shorting 1 \
  --precision_factor 2.0 \
  --auto_weight 1 \
  --query_independence

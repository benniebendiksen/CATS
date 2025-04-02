#!/bin/bash

# Check if pyenv environment is active
if [[ -z "$PYENV_VERSION" || "$PYENV_VERSION" != "3.10.16" ]]; then
  echo "Error: pyenv environment is not correctly activated."
  exit 1
fi

# Debugging: Ensure the right Python version is being used
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"

# Safety check: Verify that the correct Python version is active
active_python_version=$(python --version 2>&1)
if [[ "$active_python_version" != *"3.10.16"* ]]; then
  echo "Error: Expected Python version 3.10.16, but found $active_python_version."
  exit 1
fi

# Check if required packages are installed from requirements.txt
requirements_file="./requirements.txt"

# If requirements.txt exists and has not been installed yet, install packages
if [[ -f "$requirements_file" ]]; then
  echo "Checking if requirements are installed from $requirements_file..."

  # Install missing dependencies
  pip freeze | grep -q -f "$requirements_file" || pip install -r "$requirements_file"
else
  echo "Warning: requirements.txt not found. Skipping package installation."
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

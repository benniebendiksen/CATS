#!/bin/bash
#SBATCH --job-name=iTransformer_training
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=b.bendiksen001@umb.edu
#SBATCH -p DGXA100
##SBATCH -p Intel6240,Intel6248
#SBATCH -A pi_funda.durupinarbabur
#SBATCH --qos=scavenger
##SBATCH -w chimera12
#SBATCH -n 2                       # Number of cores
##SBATCH -N 1                       # Ensure all cores are on one machine
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --export=HOME              # Export HOME environment variable for miniconda access
#SBATCH --mem=64G                  # 64GB memory
#SBATCH -t 3-23:59:59              # near 4 days runtime
#SBATCH --output=slurm_outputs/cats_%A_%a.out
#SBATCH --error=slurm_outputs/cats_%A_%a.err
#SBATCH --array=0-1                # 0 = benchmark datasets, 1 = logits dataset

. /etc/profile

# Activate pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Set the desired Python version via pyenv
export PYENV_VERSION=3.10.16

# Safety check: Verify that the correct Python version is active
active_python_version=$(python --version 2>&1)
if [[ "$active_python_version" != *"3.10.16"* ]]; then
  echo "Error: Expected Python version 3.10.16, but found $active_python_version."
  exit 1
fi

# Activate the virtual environment
pyenv activate pyenv_gpu_env
echo "Activated pyenv environment: $(pyenv version-name)"

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Using $SLURM_CPUS_ON_NODE CPUs"
echo "Using SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Function to run benchmark datasets on GPU 0
run_benchmark_datasets() {
    # Set environment to use GPU 0
    #export CUDA_VISIBLE_DEVICES=0

    echo "============================================================"
    echo "Running benchmark datasets on GPU 0"
    echo "============================================================"


    # Navigate to the project directory
    cd /hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/CATS

    # Run the BCEWithLogitsLoss model
    echo "Running logits dataset..."
    bash ./scripts/logits_96_input.sh

    echo "logits dataset completed"

}


run_logits_dataset() {
    # Set environment to use GPU 1
    # export CUDA_VISIBLE_DEVICES=1

    echo "============================================================"
    echo "Running logits dataset on GPU 1"
    echo "============================================================"

    # Navigate to the project directory
    cd /hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/CATS

    # Run the BCEWithLogitsLoss model
    echo "Running logits_2 dataset..."
    bash ./scripts/logits_96_input_2.sh
}


# Main execution based on SLURM array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    run_benchmark_datasets
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    run_logits_dataset
else
    echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
    run_benchmark_datasets
fi

echo "Job completed at $(date)"

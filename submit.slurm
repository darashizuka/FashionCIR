#!/bin/bash
#SBATCH --job-name=fashioncir
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/sd205/fashioncir/logs/%j.out
#SBATCH --error=/scratch/sd205/fashioncir/logs/%j.err

source /scratch/sd205/fashioncir/my_env/bin/activate

export HF_HOME="/scratch/sd205/fashioncir/hf_cache"
export HF_TOKEN="__HF_TOKEN__"

SCRATCH="/scratch/sd205/fashioncir"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$SCRATCH/my_env/bin/python3 -u "$SCRATCH/train.py" \
    --data_dir "$SCRATCH/datasets/facap/facap" \
    --output_dir "$SCRATCH/output" \
    --epochs 10 \
    --batch_size 64 \
    --num_workers 8 \
    --max_samples 0

echo "Job $SLURM_JOB_ID finished at $(date)"

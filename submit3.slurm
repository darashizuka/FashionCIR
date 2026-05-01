#!/bin/bash
#SBATCH --job-name=fashioncir-ft2
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/sd205/fashioncir/logs/exp3_%j.out
#SBATCH --error=/scratch/sd205/fashioncir/logs/exp3_%j.err

source /scratch/sd205/fashioncir/my_env/bin/activate

export HF_HOME="/scratch/sd205/fashioncir/hf_cache"
export HF_TOKEN="__HF_TOKEN__"

SCRATCH="/scratch/sd205/fashioncir"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python "$SCRATCH/finetune3.py" \
    --data_dir "$SCRATCH/datasets/facap/facap" \
    --output_dir "$SCRATCH/output_finetune3" \
    --epochs 7 \
    --batch_size 32 \
    --lr 2e-6 \
    --grad_accum 2 \
    --unfreeze_vision 2 \
    --unfreeze_text 0 \
    --patience 5 \
    --num_workers 4

echo "Job $SLURM_JOB_ID finished at $(date)"

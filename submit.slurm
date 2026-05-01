#!/bin/bash
#SBATCH --job-name=fashioncir-ft
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/sd205/fashioncir/logs/%j.out
#SBATCH --error=/scratch/sd205/fashioncir/logs/%j.err

source /scratch/sd205/fashioncir/my_env/bin/activate

export HF_HOME="/scratch/sd205/fashioncir/hf_cache"
export HF_TOKEN="__HF_TOKEN__"

SCRATCH="/scratch/sd205/fashioncir"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python "$SCRATCH/finetune.py" \
    --data_dir "$SCRATCH/datasets/facap/facap" \
    --output_dir "$SCRATCH/output_finetune" \
    --checkpoint "$SCRATCH/output/best_model.pth" \
    --epochs 20 \
    --batch_size 32 \
    --fusion_lr 1e-5 \
    --clip_lr 2e-6 \
    --grad_accum 2 \
    --unfreeze_vision 2 \
    --patience 5 \
    --num_workers 4

echo "Job $SLURM_JOB_ID finished at $(date)"

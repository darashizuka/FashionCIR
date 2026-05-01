#!/bin/bash
#SBATCH --job-name=fashioncir-bl
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sd205/fashioncir/logs/baseline_%j.out
#SBATCH --error=/scratch/sd205/fashioncir/logs/baseline_%j.err

source /scratch/sd205/fashioncir/my_env/bin/activate

export HF_HOME="/scratch/sd205/fashioncir/hf_cache"

SCRATCH="/scratch/sd205/fashioncir"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

python "$SCRATCH/baseline.py" \
    --data_dir "$SCRATCH/datasets/facap/facap" \
    --output_dir "$SCRATCH/output_baseline" \
    --batch_size 128 \
    --num_workers 4

echo "Job $SLURM_JOB_ID finished at $(date)"

#!/bin/bash
#SBATCH --job-name=fashioncir-eval
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sd205/fashioncir/logs/_eval_%j.out
#SBATCH --error=/scratch/sd205/fashioncir/logs/_eval_%j.err

source /scratch/sd205/fashioncir/my_env/bin/activate

export HF_HOME="/scratch/sd205/fashioncir/hf_cache"

SCRATCH="/scratch/sd205/fashioncir"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

python "$SCRATCH/eval.py" \
    --data_dir "$SCRATCH/datasets/facap/facap" \
    --output_dir "$SCRATCH/eval_results" \
    --model_path "$SCRATCH/output_finetune/best_model.pth" \
    --num_queries 0 \
    --custom_image "/scratch/sd205/fashioncir/image.jpg" \
    --custom_text "make it short-sleeved with a polka dot pattern" \
    --top_k 5 \
    --batch_size 128 \
    --num_workers 4

echo "Job $SLURM_JOB_ID finished at $(date)"

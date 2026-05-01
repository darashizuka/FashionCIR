#!/bin/bash
# Usage: ./launch.sh [train.py]

SCRIPT_TO_RUN="${1:-train.py}"
source config.sh

WORK="/storage/hpc/work/comp646/sd205"
SCRATCH="/scratch/sd205/fashioncir"

echo "=== FashionCIR — launching $SCRIPT_TO_RUN on NOTS ==="

# --- 1. Create scratch dirs ---
echo "Creating scratch directories..."
ssh "$REMOTE" "mkdir -p $SCRATCH/logs $SCRATCH/output $SCRATCH/hf_cache"

# --- 2. Create venv DIRECTLY on scratch (GPU SAFE VERSION) ---
echo "Checking venv on scratch..."
ssh "$REMOTE" "
if [ ! -d $SCRATCH/my_env ]; then
    echo 'Creating fresh venv on scratch...'
    python3 -m venv $SCRATCH/my_env
    source $SCRATCH/my_env/bin/activate
    pip install --upgrade pip
    # Force CUDA 12.1 build even on the login node
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install open_clip_torch matplotlib pillow
    echo 'Venv created.'
else
    echo 'Venv already on scratch.'
fi
"

# --- 3. Stage datasets (Keep commented but syntax-clean) ---
# echo "Checking datasets on scratch..."
# ssh "$REMOTE" "
# if [ ! -d $SCRATCH/datasets/facap ]; then
#     mkdir -p $SCRATCH/datasets
#     cp -r $WORK/datasets/facap $SCRATCH/datasets/
# fi
# "

# --- 4. Upload script and slurm file ---
echo "Uploading files..."
sed "s|__HF_TOKEN__|$HF_TOKEN|g" eval.slurm > temp_submit.slurm
scp temp_submit.slurm "$REMOTE:$SCRATCH/eval.slurm"
scp "$SCRIPT_TO_RUN" "$REMOTE:$SCRATCH/"
rm temp_submit.slurm

# --- 5. Submit ---
echo "Submitting job..."
ssh "$REMOTE" "cd $SCRATCH && sbatch --export=ALL eval.slurm"
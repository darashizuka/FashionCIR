# FashionCIR: Composed Image Retrieval for Fashion

Composed Image Retrieval (CIR) system that takes a **reference fashion image** and a **modification text** (e.g., "make it shorter with a v-neckline") and retrieves matching target images from a catalog.

Built on [CLIP](https://github.com/mlfoundations/open_clip) (ViT-B/32) with a trainable fusion module, evaluated on the [FACap](https://github.com/fgxaos/facap-sigir25-gennext) dataset (Fashion200K images, 338K samples, 6 garment categories).


## Results

### Baselines (Zero-shot CLIP, no training)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image-only | 6.0 | 21.1 | 31.0 |
| Text-only | 6.9 | 17.9 | 24.8 |
| Arithmetic Fusion | 15.6 | 42.9 | 55.6 |

### Trained Models

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Frozen CLIP + Fusion | 26.72 | 59.04 | 72.29 |
| Partial Fine-tune (last 2 layers) + Fusion | 27.44 | 60.30 | **73.32** |
| **Caption FT + Fusion** | **27.62** | **60.43** | 73.12 |

## File Structure

### Training Scripts

| File | Description |
|---|---|
| `train.py` | Frozen CLIP + trainable fusion module (core approach) |
| `finetune.py` | End-to-end fine-tuning: unfreezes last N CLIP vision blocks + trains fusion jointly |
| `finetune3.py` | Two-stage: (1) fine-tune CLIP on image-caption pairs, (2) freeze adapted CLIP + train fusion on triplets |
| `baseline.py` | Zero-shot CLIP baselines: image-only, text-only, arithmetic fusion |
| `eval.py` | Evaluation: R@K on test set, random query visualization, custom image+text queries |

### SLURM Job Scripts

| File | Runs | Description |
|---|---|---|
| `submit_train.slurm` | `train.py` | Frozen CLIP + fusion training |
| `submit.slurm` | `finetune.py` | End-to-end fine-tuning |
| `submit3.slurm` | `finetune3.py` | Caption fine-tuning + fusion |
| `baseline.slurm` | `baseline.py` | Zero-shot baselines |
| `eval.slurm` | `eval.py` | Evaluation with frozen CLIP model |
| `caption_eval.slurm` | `eval.py` | Evaluation with caption-adapted CLIP model |
| `launch.sh` | — | Utility to upload files and submit jobs to cluster |

### Other Files

| File | Description |
|---|---|
| `finetune2.py` | Earlier variant of `finetune.py` (kept for reference) |
| `caption_eval.py` | Copy of `eval.py` configured for caption-FT evaluation |

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch matplotlib numpy pillow
```

## Dataset

Download the [FACap dataset](https://github.com/layer6ai-labs/FACap):

```
data_dir/
  f200k_images/         # Fashion200K images
  cir_triplets/         # (reference, text, target) triplet JSONs per category
  image_captions/       # Image-caption pair JSONs per category
```

## Usage

### Train (Frozen CLIP + Fusion)

```bash
python train.py \
    --data_dir path/to/facap \
    --output_dir output_train \
    --epochs 10 \
    --batch_size 64
```

### Baselines

```bash
python baseline.py \
    --data_dir path/to/facap \
    --output_dir output_baseline
```

### Fine-tune (End-to-End)

Requires a pretrained fusion checkpoint from `train.py`:

```bash
python finetune.py \
    --data_dir path/to/facap \
    --output_dir output_finetune \
    --checkpoint output_train/best_model.pth \
    --unfreeze_vision 2 \
    --fusion_lr 1e-5 \
    --clip_lr 2e-6
```

### Caption Fine-tuning (Two-Stage)

```bash
python finetune3.py \
    --data_dir path/to/facap \
    --output_dir output_caption_ft \
    --epochs 7 \
    --lr 2e-6 \
    --unfreeze_vision 2 \
    --unfreeze_text 0
```

### Evaluate

```bash
# R@K on test set + qualitative results
python eval.py \
    --data_dir path/to/facap \
    --model_path output_train/best_model.pth \
    --output_dir eval_results \
    --num_queries 20

# With caption-adapted CLIP
python eval.py \
    --data_dir path/to/facap \
    --model_path output_caption_ft/triplet_results/best_model.pth \
    --clip_checkpoint output_caption_ft/best_clip.pth \
    --output_dir eval_results

# Custom query
python eval.py \
    --data_dir path/to/facap \
    --model_path output_train/best_model.pth \
    --output_dir eval_results \
    --custom_image your_image.jpg \
    --custom_text "make it shorter with a floral pattern" \
    --num_queries 0
```

## Data Split

70% train / 15% val / 15% test, stratified per garment category (seed=42). All scripts use the same split for comparability.

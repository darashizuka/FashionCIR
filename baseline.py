import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import open_clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FACaPDataset(Dataset):
    def __init__(self, triplet_file, image_base, preprocess, tokenizer):
        self.image_base = image_base
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.triplets = []
        self.category = os.path.basename(triplet_file).replace("_train_triplets.json", "")

        with open(triplet_file) as f:
            data = json.load(f)

        skipped = 0
        for item in data:
            ref_path = os.path.join(self.image_base, item["candidate"])
            tar_path = os.path.join(self.image_base, item["target"])
            if os.path.exists(ref_path) and os.path.exists(tar_path):
                self.triplets.append(item)
            else:
                skipped += 1

        print(f"  {self.category}: {len(self.triplets)} valid  |  {skipped} skipped")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        ref_img = self.preprocess(Image.open(os.path.join(self.image_base, item["candidate"])).convert("RGB"))
        tar_img = self.preprocess(Image.open(os.path.join(self.image_base, item["target"])).convert("RGB"))
        text_tok = self.tokenizer(item["captions"][0])
        return ref_img, text_tok.squeeze(0), tar_img, idx


def load_test_set(data_dir, preprocess, tokenizer, seed=42):
    """Load only the test split (same 70/15/15 split as train.py)."""
    triplet_dir = os.path.join(data_dir, "cir_triplets")
    json_files = sorted(f for f in os.listdir(triplet_dir) if f.endswith(".json"))

    test_sets = []
    rng = np.random.RandomState(seed)

    print("Loading test split (70/15/15 per category)...")
    for jf in json_files:
        cat_ds = FACaPDataset(os.path.join(triplet_dir, jf), data_dir, preprocess, tokenizer)
        n = len(cat_ds)
        indices = rng.permutation(n)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        test_idx = indices[n_train + n_val:].tolist()
        test_sets.append(Subset(cat_ds, test_idx))

    test_ds = ConcatDataset(test_sets)
    print(f"Test set: {len(test_ds)} triplets")
    return test_ds


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_all_features(clip_model, loader, device):
    """Pre-compute all CLIP features for the test set."""
    all_ref, all_txt, all_tgt = [], [], []

    for ref_img, text, tar_img, _ in loader:
        ref_feat = F.normalize(clip_model.encode_image(ref_img.to(device)).float(), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(text.to(device)).float(), dim=-1)
        tgt_feat = F.normalize(clip_model.encode_image(tar_img.to(device)).float(), dim=-1)
        all_ref.append(ref_feat)
        all_txt.append(text_feat)
        all_tgt.append(tgt_feat)

    return torch.cat(all_ref), torch.cat(all_txt), torch.cat(all_tgt)


def recall_at_k(sim_matrix, ks=(1, 5, 10)):
    device = sim_matrix.device
    labels = torch.arange(len(sim_matrix), device=device)
    results = {}
    for k in ks:
        topk = sim_matrix.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item() * 100
    return results


def run_baselines(all_ref, all_txt, all_tgt):
    results = {}

    # 1. Image-only: use reference image embedding to find target
    sim_img = all_ref @ all_tgt.t()
    results["Image-only"] = recall_at_k(sim_img)

    # 2. Text-only: use modification text embedding to find target
    sim_txt = all_txt @ all_tgt.t()
    results["Text-only"] = recall_at_k(sim_txt)

    # 3. Arithmetic fusion: q = normalize(f_v + f_t)
    fused = F.normalize(all_ref + all_txt, dim=-1)
    sim_arith = fused @ all_tgt.t()
    results["Arithmetic"] = recall_at_k(sim_arith)

    return results


# ---------------------------------------------------------------------------
# Qualitative results for arithmetic baseline
# ---------------------------------------------------------------------------
def _get_triplet_at(dataset, flat_idx):
    if isinstance(dataset, FACaPDataset):
        return dataset.triplets[flat_idx]
    elif isinstance(dataset, Subset):
        real_idx = dataset.indices[flat_idx]
        return _get_triplet_at(dataset.dataset, real_idx)
    elif isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if flat_idx < len(ds):
                return _get_triplet_at(ds, flat_idx)
            flat_idx -= len(ds)
    return None


def _get_image_base(dataset):
    if isinstance(dataset, FACaPDataset):
        return dataset.image_base
    elif isinstance(dataset, Subset):
        return _get_image_base(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        return _get_image_base(dataset.datasets[0])
    return None


def save_qualitative_results(all_pred, all_tgt, dataset, output_dir,
                             num_queries=20, top_k=5):
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    image_base = _get_image_base(dataset)
    n = len(all_pred)
    chosen = np.random.RandomState(456).choice(n, size=min(num_queries, n), replace=False)

    for qi, pos in enumerate(chosen):
        sims = all_pred[pos] @ all_tgt.t()
        topk_positions = sims.topk(top_k).indices.tolist()

        triplet = _get_triplet_at(dataset, int(pos))
        if triplet is None:
            continue

        ref_path = os.path.join(image_base, triplet["candidate"])
        gt_path = os.path.join(image_base, triplet["target"])
        caption = triplet["captions"][0]

        fig, axes = plt.subplots(1, 2 + top_k, figsize=(3 * (2 + top_k), 4))
        axes[0].imshow(Image.open(ref_path).convert("RGB"))
        axes[0].set_title("Reference", fontsize=9); axes[0].axis("off")
        axes[1].imshow(Image.open(gt_path).convert("RGB"))
        axes[1].set_title("Ground Truth", fontsize=9); axes[1].axis("off")

        for r, ret_pos in enumerate(topk_positions):
            ret_triplet = _get_triplet_at(dataset, int(ret_pos))
            if ret_triplet is None:
                continue
            ret_path = os.path.join(image_base, ret_triplet["target"])
            axes[2 + r].imshow(Image.open(ret_path).convert("RGB"))
            is_correct = (ret_pos == pos)
            color = "green" if is_correct else "gray"
            axes[2 + r].set_title(f"Rank {r+1}" + (" *" if is_correct else ""),
                                  fontsize=9, color=color)
            axes[2 + r].axis("off")

        fig.suptitle(f'"{caption[:100]}"', fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(qual_dir, f"query_{qi:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(chosen)} qualitative results to {qual_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FashionCIR Baselines (zero-shot CLIP)")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sd205/fashioncir/datasets/facap/facap")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sd205/fashioncir/output_baseline")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- CLIP (frozen, pretrained) ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device).eval()

    # --- Test set (same split as train.py) ---
    test_ds = load_test_set(args.data_dir, preprocess, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Compute features ---
    print("Computing CLIP features...")
    all_ref, all_txt, all_tgt = compute_all_features(clip_model, test_loader, device)

    # --- Run all baselines ---
    print("\n" + "="*60)
    print("BASELINE RESULTS (zero-shot CLIP, no training)")
    print("="*60)

    results = run_baselines(all_ref, all_txt, all_tgt)

    for method, metrics in results.items():
        print(f"  {method:20s}  R@1={metrics['R@1']:6.2f}  R@5={metrics['R@5']:6.2f}  R@10={metrics['R@10']:6.2f}")

    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Bar chart comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(results.keys())
    x = np.arange(3)
    width = 0.25

    for i, method in enumerate(methods):
        vals = [results[method]["R@1"], results[method]["R@5"], results[method]["R@10"]]
        bars = ax.bar(x + i * width, vals, width, label=method)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Baseline Comparison (Zero-shot CLIP)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["R@1", "R@5", "R@10"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "baseline_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {args.output_dir}/baseline_comparison.png")

    # --- Qualitative results for arithmetic baseline ---
    fused = F.normalize(all_ref + all_txt, dim=-1)
    save_qualitative_results(fused.cpu(), all_tgt.cpu(), test_ds, args.output_dir,
                             num_queries=20, top_k=5)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()

import os
import json
import argparse
import torch
import torch.nn as nn
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

    return ConcatDataset(test_sets)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FashionFusionModule(nn.Module):
    def __init__(self, clip_dim=512):
        super().__init__()
        self.combiner = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim * 2),
            nn.BatchNorm1d(clip_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(clip_dim * 2, clip_dim),
            nn.BatchNorm1d(clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ref_features, text_features):
        ref_features = F.normalize(ref_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        combined = torch.cat([ref_features, text_features], dim=-1)
        fused = self.combiner(combined)
        return F.normalize(fused, dim=-1)


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Pre-compute target embedding bank
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_target_bank(clip_model, dataset, device, batch_size=128, num_workers=4):
    """Encode all target images in the test set into an embedding bank."""
    clip_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_tgt = []
    for _, _, tar_img, _ in loader:
        feat = F.normalize(clip_model.encode_image(tar_img.to(device)).float(), dim=-1)
        all_tgt.append(feat)
    return torch.cat(all_tgt)


# ---------------------------------------------------------------------------
# Mode 1: Random samples from test set (with ground truth)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_random(model, clip_model, dataset, target_bank, device, output_dir,
                num_queries=50, top_k=5, seed=789):
    model.eval()
    clip_model.eval()
    qual_dir = os.path.join(output_dir, "random_queries")
    os.makedirs(qual_dir, exist_ok=True)

    image_base = _get_image_base(dataset)
    n = len(dataset)
    chosen = np.random.RandomState(seed).choice(n, size=min(num_queries, n), replace=False)

    # Compute query embeddings for chosen samples
    results_log = []
    for qi, pos in enumerate(chosen):
        triplet = _get_triplet_at(dataset, int(pos))
        if triplet is None:
            continue

        ref_path = os.path.join(image_base, triplet["candidate"])
        gt_path = os.path.join(image_base, triplet["target"])
        caption = triplet["captions"][0]

        # Get dataset item (preprocessed)
        ref_img, text_tok, _, _ = dataset[int(pos)]
        ref_img = ref_img.unsqueeze(0).to(device)
        text_tok = text_tok.unsqueeze(0).to(device)

        ref_feat = clip_model.encode_image(ref_img).float()
        text_feat = clip_model.encode_text(text_tok).float()
        query_emb = model(ref_feat, text_feat)

        sims = (query_emb @ target_bank.t()).squeeze(0)
        topk_vals, topk_idx = sims.topk(top_k)

        # Check if ground truth is in top-K
        gt_feat = F.normalize(clip_model.encode_image(
            dataset[int(pos)][2].unsqueeze(0).to(device)
        ).float(), dim=-1)
        gt_sim = (query_emb @ gt_feat.t()).item()
        gt_rank = (sims >= gt_sim).sum().item()

        results_log.append({
            "query_idx": int(pos),
            "caption": caption,
            "gt_rank": gt_rank,
            "top_k_sims": topk_vals.cpu().tolist(),
        })

        # Draw
        fig, axes = plt.subplots(1, 2 + top_k, figsize=(3 * (2 + top_k), 4))

        axes[0].imshow(Image.open(ref_path).convert("RGB"))
        axes[0].set_title("Reference", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(Image.open(gt_path).convert("RGB"))
        axes[1].set_title(f"GT (rank {gt_rank})", fontsize=9,
                          color="green" if gt_rank <= top_k else "red")
        axes[1].axis("off")

        for r, ret_pos in enumerate(topk_idx.tolist()):
            ret_triplet = _get_triplet_at(dataset, ret_pos)
            if ret_triplet is None:
                continue
            ret_path = os.path.join(image_base, ret_triplet["target"])
            axes[2 + r].imshow(Image.open(ret_path).convert("RGB"))
            is_gt = (ret_pos == pos)
            color = "green" if is_gt else "gray"
            axes[2 + r].set_title(
                f"Rank {r+1} ({topk_vals[r]:.3f})" + (" *" if is_gt else ""),
                fontsize=8, color=color
            )
            axes[2 + r].axis("off")

        fig.suptitle(f'"{caption[:120]}"', fontsize=9, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(qual_dir, f"query_{qi:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(qual_dir, "query_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"Saved {len(results_log)} random query results to {qual_dir}/")


# ---------------------------------------------------------------------------
# Mode 2: Custom query (your own image + text)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_custom(model, clip_model, preprocess, tokenizer, target_bank,
                dataset, device, image_path, text_query, output_dir, top_k=5):
    model.eval()
    clip_model.eval()
    image_base = _get_image_base(dataset)

    ref_img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tok = tokenizer(text_query).to(device)

    ref_feat = clip_model.encode_image(ref_img).float()
    text_feat = clip_model.encode_text(text_tok).float()
    query_emb = model(ref_feat, text_feat)

    sims = (query_emb @ target_bank.t()).squeeze(0)
    topk_vals, topk_idx = sims.topk(top_k)

    fig, axes = plt.subplots(1, 1 + top_k, figsize=(3 * (1 + top_k), 4))

    axes[0].imshow(Image.open(image_path).convert("RGB"))
    axes[0].set_title("Your Query Image", fontsize=9)
    axes[0].axis("off")

    for r, ret_pos in enumerate(topk_idx.tolist()):
        ret_triplet = _get_triplet_at(dataset, ret_pos)
        if ret_triplet is None:
            continue
        ret_path = os.path.join(image_base, ret_triplet["target"])
        axes[1 + r].imshow(Image.open(ret_path).convert("RGB"))
        axes[1 + r].set_title(f"Rank {r+1} ({topk_vals[r]:.3f})", fontsize=8)
        axes[1 + r].axis("off")

    fig.suptitle(f'"{text_query}"', fontsize=10, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "custom_query.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Custom query result saved to {out_path}")


# ---------------------------------------------------------------------------
# R@K on full test set
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_recall(model, clip_model, dataset, target_bank, device,
                batch_size=64, num_workers=4, ks=(1, 5, 10)):
    model.eval()
    clip_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_pred = []
    for ref_img, text, _, _ in loader:
        ref_feat = clip_model.encode_image(ref_img.to(device)).float()
        text_feat = clip_model.encode_text(text.to(device)).float()
        predicted = model(ref_feat, text_feat)
        all_pred.append(predicted)

    all_pred = torch.cat(all_pred)
    sim = all_pred @ target_bank.t()

    results = {}
    labels = torch.arange(len(sim), device=device)
    for k in ks:
        topk = sim.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item() * 100
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FashionCIR Evaluation")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sd205/fashioncir/datasets/facap/facap")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sd205/fashioncir/eval_results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth (fusion module checkpoint)")
    parser.add_argument("--clip_checkpoint", type=str, default="",
                        help="Path to caption-finetuned CLIP (from finetune3.py). Leave empty for default CLIP.")

    parser.add_argument("--num_queries", type=int, default=50,
                        help="Number of random test queries to visualize")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=789,
                        help="Random seed for selecting test queries")

    # Custom query mode
    parser.add_argument("--custom_image", type=str, default="",
                        help="Path to your own reference image")
    parser.add_argument("--custom_text", type=str, default="",
                        help="Your modification text query")

    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- CLIP ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    if args.clip_checkpoint and os.path.exists(args.clip_checkpoint):
        print(f"Loading caption-finetuned CLIP: {args.clip_checkpoint}")
        ckpt = torch.load(args.clip_checkpoint, map_location=device, weights_only=False)
        clip_model.load_state_dict(ckpt["clip_state_dict"])

    clip_model = clip_model.to(device).eval()

    # --- Fusion model ---
    clip_dim = 512 if "B-32" in args.clip_model or "B-16" in args.clip_model else 768
    model = FashionFusionModule(clip_dim=clip_dim).to(device)

    print(f"Loading fusion model: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "metrics" in ckpt:
        print(f"  Checkpoint metrics: {ckpt['metrics']}")
    model.eval()

    # --- Test set ---
    test_ds = load_test_set(args.data_dir, preprocess, tokenizer)
    print(f"Test set: {len(test_ds)} triplets")

    # --- Build target embedding bank ---
    print("Building target embedding bank...")
    target_bank = build_target_bank(clip_model, test_ds, device,
                                     args.batch_size, args.num_workers)
    print(f"Target bank: {target_bank.shape}")

    # --- R@K on full test set ---
    print("\nComputing R@K on full test set...")
    recall = eval_recall(model, clip_model, test_ds, target_bank, device,
                         args.batch_size, args.num_workers)
    print(f"TEST — R@1={recall['R@1']:.2f}  R@5={recall['R@5']:.2f}  R@10={recall['R@10']:.2f}")

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(recall, f, indent=2)

    # --- Random queries from test set ---
    print(f"\nGenerating {args.num_queries} random query visualizations...")
    eval_random(model, clip_model, test_ds, target_bank, device, args.output_dir,
                num_queries=args.num_queries, top_k=args.top_k, seed=args.seed)

    # --- Custom query (if provided) ---
    if args.custom_image and args.custom_text:
        print(f"\nRunning custom query: '{args.custom_text}' on {args.custom_image}")
        eval_custom(model, clip_model, preprocess, tokenizer, target_bank,
                    test_ds, device, args.custom_image, args.custom_text,
                    args.output_dir, top_k=args.top_k)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()

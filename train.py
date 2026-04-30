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


def load_and_split(data_dir, preprocess, tokenizer, train_ratio=0.70, val_ratio=0.15,
                   max_samples=0, seed=42):
    """Load each category file separately, split 70/15/15 per category, then combine."""
    triplet_dir = os.path.join(data_dir, "cir_triplets")
    json_files = sorted(f for f in os.listdir(triplet_dir) if f.endswith(".json"))

    train_sets, val_sets, test_sets = [], [], []
    rng = np.random.RandomState(seed)

    print(f"Loading triplets with {int(train_ratio*100)}/{int(val_ratio*100)}/{int((1-train_ratio-val_ratio)*100)} split per category...")
    for jf in json_files:
        cat_ds = FACaPDataset(os.path.join(triplet_dir, jf), data_dir, preprocess, tokenizer)
        n = len(cat_ds)
        indices = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()

        train_sets.append(Subset(cat_ds, train_idx))
        val_sets.append(Subset(cat_ds, val_idx))
        test_sets.append(Subset(cat_ds, test_idx))

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    test_ds = ConcatDataset(test_sets)

    if max_samples > 0:
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max_samples // 5, len(val_ds))))
        test_ds = Subset(test_ds, range(min(max_samples // 5, len(test_ds))))

    print(f"Total — Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


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
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, clip_model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (ref_img, text, tar_img, _) in enumerate(loader):
        ref_img = ref_img.to(device)
        text = text.to(device)
        tar_img = tar_img.to(device)

        with torch.no_grad():
            ref_feat = clip_model.encode_image(ref_img).float()
            text_feat = clip_model.encode_text(text).float()
            target_feat = F.normalize(clip_model.encode_image(tar_img).float(), dim=-1)

        predicted = model(ref_feat, text_feat)
        logits = model.logit_scale.exp() * (predicted @ target_feat.t())
        labels = torch.arange(len(logits), device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation — Recall@K
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, clip_model, loader, device, ks=(1, 5, 10)):
    model.eval()
    all_pred = []
    all_tgt = []

    for ref_img, text, tar_img, _ in loader:
        ref_img = ref_img.to(device)
        text = text.to(device)
        tar_img = tar_img.to(device)

        ref_feat = clip_model.encode_image(ref_img).float()
        text_feat = clip_model.encode_text(text).float()
        target_feat = F.normalize(clip_model.encode_image(tar_img).float(), dim=-1)

        predicted = model(ref_feat, text_feat)
        all_pred.append(predicted)
        all_tgt.append(target_feat)

    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    sim = all_pred @ all_tgt.t()

    results = {}
    for k in ks:
        topk = sim.topk(k, dim=1).indices
        correct = (topk == torch.arange(len(sim), device=device).unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item() * 100
    return results


# ---------------------------------------------------------------------------
# Qualitative results — save image grids
# ---------------------------------------------------------------------------
def get_base_dataset_and_triplets(dataset):
    """Unwrap ConcatDataset/Subset layers to get the underlying FACaPDataset and global triplet list."""
    all_triplets = []
    image_base = None

    def collect(ds):
        nonlocal image_base
        if isinstance(ds, FACaPDataset):
            if image_base is None:
                image_base = ds.image_base
            all_triplets.extend(ds.triplets)
        elif isinstance(ds, Subset):
            collect(ds.dataset)
        elif isinstance(ds, ConcatDataset):
            for d in ds.datasets:
                collect(d)

    collect(dataset)
    return all_triplets, image_base


@torch.no_grad()
def save_qualitative_results(model, clip_model, dataset, device, output_dir,
                             num_queries=20, top_k=5):
    model.eval()
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    all_triplets, image_base = get_base_dataset_and_triplets(dataset)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    all_pred = []
    all_tgt = []
    for ref_img, text, tar_img, _ in loader:
        ref_feat = clip_model.encode_image(ref_img.to(device)).float()
        text_feat = clip_model.encode_text(text.to(device)).float()
        target_feat = F.normalize(clip_model.encode_image(tar_img.to(device)).float(), dim=-1)
        predicted = model(ref_feat, text_feat)
        all_pred.append(predicted.cpu())
        all_tgt.append(target_feat.cpu())

    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)

    n = len(all_pred)
    chosen = np.random.RandomState(123).choice(n, size=min(num_queries, n), replace=False)

    for qi, pos in enumerate(chosen):
        sims = all_pred[pos] @ all_tgt.t()
        topk_positions = sims.topk(top_k).indices.tolist()

        # Map flat position back to the underlying triplet
        # Walk the ConcatDataset to find which sub-dataset and index
        triplet = _get_triplet_at(dataset, int(pos))
        if triplet is None:
            continue

        ref_path = os.path.join(image_base, triplet["candidate"])
        gt_path = os.path.join(image_base, triplet["target"])
        caption = triplet["captions"][0]

        fig, axes = plt.subplots(1, 2 + top_k, figsize=(3 * (2 + top_k), 4))

        axes[0].imshow(Image.open(ref_path).convert("RGB"))
        axes[0].set_title("Reference", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(Image.open(gt_path).convert("RGB"))
        axes[1].set_title("Ground Truth", fontsize=9)
        axes[1].axis("off")

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


def _get_triplet_at(dataset, flat_idx):
    """Resolve a flat index through ConcatDataset/Subset to the underlying triplet dict."""
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


# ---------------------------------------------------------------------------
# Plot R@K over epochs
# ---------------------------------------------------------------------------
def plot_metrics(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["loss"] for h in history], "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    for key in ["R@1", "R@5", "R@10"]:
        ax2.plot(epochs, [h[key] for h in history], "-o", markersize=3, label=key)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall (%)")
    ax2.set_title("Recall@K on Validation Set")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_dir}/training_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FashionCIR Training")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sd205/fashioncir/datasets/facap/facap")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sd205/fashioncir/output")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit dataset size for debugging (0 = use all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- CLIP backbone (frozen) ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # --- Dataset: 70/15/15 stratified per category ---
    train_ds, val_ds, test_ds = load_and_split(
        args.data_dir, preprocess, tokenizer, max_samples=args.max_samples
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Fusion model ---
    clip_dim = 512 if "B-32" in args.clip_model or "B-16" in args.clip_model else 768
    model = FashionFusionModule(clip_dim=clip_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    history = []
    best_r10 = 0.0
    start_epoch = 1

    resume_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    if os.path.exists(resume_path):
        print(f"--- Found checkpoint at {resume_path}. Resuming... ---")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        
        # Restore history and best_r10 if they exist
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                history = json.load(f)
            best_r10 = max([h["R@10"] for h in history]) if history else 0.0
        
        # Fast-forward the scheduler to the correct epoch
        for _ in range(1, start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, clip_model, train_loader, optimizer, device, epoch)
        scheduler.step()

        metrics = evaluate(model, clip_model, val_loader, device)
        entry = {"epoch": epoch, "loss": avg_loss, **metrics}
        history.append(entry)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"=== Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"R@1={metrics['R@1']:.2f}  R@5={metrics['R@5']:.2f}  R@10={metrics['R@10']:.2f} ===")

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  -> New best model saved (R@10={best_r10:.2f})")

        torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        }, os.path.join(args.output_dir, "latest_checkpoint.pth"))

    # --- Final model ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(args.output_dir, "final_model.pth"))

    # --- Plot training curves ---
    plot_metrics(history, args.output_dir)

    # --- Final evaluation on TEST set ---
    print("\n=== Final evaluation on held-out TEST set ===")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, clip_model, test_loader, device)
    print(f"TEST — R@1={test_metrics['R@1']:.2f}  R@5={test_metrics['R@5']:.2f}  R@10={test_metrics['R@10']:.2f}")

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # --- Qualitative results on test set ---
    save_qualitative_results(model, clip_model, test_ds, device, args.output_dir,
                             num_queries=20, top_k=5)

    print(f"\nDone. All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

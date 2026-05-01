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
# Dataset (same as train.py)
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

        train_sets.append(Subset(cat_ds, indices[:n_train].tolist()))
        val_sets.append(Subset(cat_ds, indices[n_train:n_train + n_val].tolist()))
        test_sets.append(Subset(cat_ds, indices[n_train + n_val:].tolist()))

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
# Model (same architecture)
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
# Unfreeze last N layers of CLIP vision encoder
# ---------------------------------------------------------------------------
def unfreeze_clip_layers(clip_model, unfreeze_vision_layers=2, unfreeze_text_layers=0):
    """Unfreeze the last N transformer blocks of CLIP's vision/text encoders."""
    for p in clip_model.parameters():
        p.requires_grad = False

    # Vision encoder — unfreeze last N blocks
    if unfreeze_vision_layers > 0:
        vision_blocks = clip_model.visual.transformer.resblocks
        total = len(vision_blocks)
        for i in range(total - unfreeze_vision_layers, total):
            for p in vision_blocks[i].parameters():
                p.requires_grad = True
        # Also unfreeze the final layer norm
        if hasattr(clip_model.visual, 'ln_post'):
            for p in clip_model.visual.ln_post.parameters():
                p.requires_grad = True
        print(f"  Vision: unfroze last {unfreeze_vision_layers}/{total} blocks")

    # Text encoder — unfreeze last N blocks
    if unfreeze_text_layers > 0:
        text_blocks = clip_model.transformer.resblocks
        total = len(text_blocks)
        for i in range(total - unfreeze_text_layers, total):
            for p in text_blocks[i].parameters():
                p.requires_grad = True
        if hasattr(clip_model, 'ln_final'):
            for p in clip_model.ln_final.parameters():
                p.requires_grad = True
        print(f"  Text: unfroze last {unfreeze_text_layers}/{total} blocks")

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in clip_model.parameters())
    print(f"  CLIP trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Training — now with gradients through CLIP
# ---------------------------------------------------------------------------
def train_one_epoch(model, clip_model, loader, optimizer, device, epoch, grad_accum=1):
    model.train()
    clip_model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (ref_img, text, tar_img, _) in enumerate(loader):
        ref_img = ref_img.to(device)
        text = text.to(device)
        tar_img = tar_img.to(device)

        ref_feat = clip_model.encode_image(ref_img).float()
        text_feat = clip_model.encode_text(text).float()

        with torch.no_grad():
            target_feat = F.normalize(clip_model.encode_image(tar_img).float(), dim=-1)

        predicted = model(ref_feat, text_feat)
        logits = model.logit_scale.exp() * (predicted @ target_feat.t())
        labels = torch.arange(len(logits), device=device)
        loss = F.cross_entropy(logits, labels) / grad_accum

        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(clip_model.parameters()), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        if batch_idx % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)}  loss={loss.item()*grad_accum:.4f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, clip_model, loader, device, ks=(1, 5, 10)):
    model.eval()
    clip_model.eval()
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
# Qualitative results
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


@torch.no_grad()
def save_qualitative_results(model, clip_model, dataset, device, output_dir,
                             num_queries=20, top_k=5):
    model.eval()
    clip_model.eval()
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    image_base = _get_image_base(dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    all_pred, all_tgt = [], []

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


# ---------------------------------------------------------------------------
# Plot metrics
# ---------------------------------------------------------------------------
def plot_metrics(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["loss"] for h in history], "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Fine-tune Training Loss")
    ax1.grid(True, alpha=0.3)

    for key in ["R@1", "R@5", "R@10"]:
        ax2.plot(epochs, [h[key] for h in history], "-o", markersize=3, label=key)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall (%)")
    ax2.set_title("Fine-tune Recall@K (Val)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "finetune_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Curves saved to {output_dir}/finetune_curves.png")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.counter = 0

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping: no improvement in R@10 for {self.patience} epochs "
                  f"(best={self.best:.2f}, current={metric:.2f})")
            return True
        return False


# ---------------------------------------------------------------------------
# Save / load checkpoint (full state for resume)
# ---------------------------------------------------------------------------
def save_checkpoint(path, epoch, model, clip_model, optimizer, scheduler, history, best_r10):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "clip_state_dict": clip_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_r10": best_r10,
    }, path)


def load_checkpoint(path, model, clip_model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["history"], ckpt["best_r10"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FashionCIR Fine-tuning (CLIP unfrozen)")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sd205/fashioncir/datasets/facap/facap")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sd205/fashioncir/output_finetune")
    parser.add_argument("--checkpoint", type=str,
                        default="/scratch/sd205/fashioncir/output/best_model.pth",
                        help="Path to frozen-training checkpoint to initialize from")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to finetune checkpoint to resume from")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fusion_lr", type=float, default=1e-5,
                        help="LR for fusion module (lower than frozen training)")
    parser.add_argument("--clip_lr", type=float, default=2e-6,
                        help="LR for unfrozen CLIP layers (much lower)")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (saves GPU memory)")
    parser.add_argument("--unfreeze_vision", type=int, default=2,
                        help="Number of CLIP vision transformer blocks to unfreeze")
    parser.add_argument("--unfreeze_text", type=int, default=0,
                        help="Number of CLIP text transformer blocks to unfreeze")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")

    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Fine-tuning config: fusion_lr={args.fusion_lr}, clip_lr={args.clip_lr}, "
          f"unfreeze_vision={args.unfreeze_vision}, unfreeze_text={args.unfreeze_text}")

    # --- CLIP backbone ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device)

    # --- Fusion model ---
    clip_dim = 512 if "B-32" in args.clip_model or "B-16" in args.clip_model else 768
    model = FashionFusionModule(clip_dim=clip_dim).to(device)

    # --- Load frozen-training checkpoint into fusion module ---
    if not args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading frozen-training checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "metrics" in ckpt:
            print(f"  Frozen-training metrics: {ckpt['metrics']}")

    # --- Unfreeze selected CLIP layers ---
    unfreeze_clip_layers(clip_model, args.unfreeze_vision, args.unfreeze_text)

    # --- Optimizer: differential LR ---
    clip_params = [p for p in clip_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.fusion_lr},
        {"params": clip_params, "lr": args.clip_lr, "weight_decay": 1e-2},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Resume from finetune checkpoint ---
    start_epoch = 1
    history = []
    best_r10 = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        start_epoch, history, best_r10 = load_checkpoint(
            args.resume, model, clip_model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best R@10={best_r10:.2f}")

    # --- Dataset (same split as train.py — same seed) ---
    train_ds, val_ds, test_ds = load_and_split(
        args.data_dir, preprocess, tokenizer, max_samples=args.max_samples
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Training ---
    early_stop = EarlyStopping(patience=args.patience)

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, clip_model, train_loader, optimizer, device,
                                   epoch, args.grad_accum)
        scheduler.step()

        metrics = evaluate(model, clip_model, val_loader, device)
        entry = {"epoch": epoch, "loss": avg_loss, **metrics}
        history.append(entry)

        with open(os.path.join(args.output_dir, "finetune_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"=== Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"R@1={metrics['R@1']:.2f}  R@5={metrics['R@5']:.2f}  R@10={metrics['R@10']:.2f} ===")

        # Save best model
        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "clip_state_dict": clip_model.state_dict(),
                "metrics": metrics,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  -> New best model (R@10={best_r10:.2f})")

        # Save resume checkpoint every epoch (overwritten each time)
        save_checkpoint(os.path.join(args.output_dir, "resume_ckpt.pth"),
                        epoch, model, clip_model, optimizer, scheduler, history, best_r10)

        # Early stopping
        if early_stop.step(metrics["R@10"]):
            break

    # --- Plot ---
    plot_metrics(history, args.output_dir)

    # --- Final test eval with best model ---
    print("\n=== Final evaluation on held-out TEST set ===")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    clip_model.load_state_dict(best_ckpt["clip_state_dict"])

    test_metrics = evaluate(model, clip_model, test_loader, device)
    print(f"TEST — R@1={test_metrics['R@1']:.2f}  R@5={test_metrics['R@5']:.2f}  R@10={test_metrics['R@10']:.2f}")

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # --- Qualitative results ---
    save_qualitative_results(model, clip_model, test_ds, device, args.output_dir,
                             num_queries=20, top_k=5)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()

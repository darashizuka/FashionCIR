import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import ConcatDataset
import open_clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset — image-caption pairs from image_captions/ JSONs
# ---------------------------------------------------------------------------
class FashionCaptionDataset(Dataset):
    def __init__(self, captions_dir, image_base, preprocess, tokenizer):
        self.image_base = image_base
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.pairs = []

        json_files = sorted(f for f in os.listdir(captions_dir) if f.endswith(".json"))
        print(f"Loading image-caption pairs from {len(json_files)} files...")

        skipped = 0
        for jf in json_files:
            category = jf.replace("_train_captions.json", "")
            with open(os.path.join(captions_dir, jf)) as f:
                data = json.load(f)
            for img_path, caption in data.items():
                full_path = os.path.join(self.image_base, img_path)
                if os.path.exists(full_path):
                    self.pairs.append({"image": img_path, "caption": caption})
                else:
                    skipped += 1
            print(f"  {category}: {len(data)} entries")

        print(f"Total valid pairs: {len(self.pairs)}  |  Skipped: {skipped}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = self.preprocess(
            Image.open(os.path.join(self.image_base, pair["image"])).convert("RGB")
        )
        text = self.tokenizer(pair["caption"])
        return image, text.squeeze(0)


# ---------------------------------------------------------------------------
# Triplet dataset (for Stage 2)
# ---------------------------------------------------------------------------
class FACaPTripletDataset(Dataset):
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


# ---------------------------------------------------------------------------
# Fusion model (for Stage 2)
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
# Triplet evaluation — R@K (for Stage 2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_triplets(model, clip_model, loader, device, ks=(1, 5, 10)):
    model.eval()
    clip_model.eval()
    all_pred, all_tgt = [], []

    for ref_img, text, tar_img, _ in loader:
        ref_feat = clip_model.encode_image(ref_img.to(device)).float()
        text_feat = clip_model.encode_text(text.to(device)).float()
        target_feat = F.normalize(clip_model.encode_image(tar_img.to(device)).float(), dim=-1)
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
# Qualitative results (for Stage 2)
# ---------------------------------------------------------------------------
def _get_triplet_at(dataset, flat_idx):
    if isinstance(dataset, FACaPTripletDataset):
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
    if isinstance(dataset, FACaPTripletDataset):
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
# CLIP contrastive loss (symmetric)
# ---------------------------------------------------------------------------
def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * (image_features @ text_features.t())
    logits_per_text = logits_per_image.t()

    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------------
# Unfreeze CLIP layers
# ---------------------------------------------------------------------------
def unfreeze_clip_layers(clip_model, unfreeze_vision_layers=4, unfreeze_text_layers=2):
    for p in clip_model.parameters():
        p.requires_grad = False

    if unfreeze_vision_layers > 0:
        vision_blocks = clip_model.visual.transformer.resblocks
        total = len(vision_blocks)
        for i in range(total - unfreeze_vision_layers, total):
            for p in vision_blocks[i].parameters():
                p.requires_grad = True
        if hasattr(clip_model.visual, 'ln_post'):
            for p in clip_model.visual.ln_post.parameters():
                p.requires_grad = True
        print(f"  Vision: unfroze last {unfreeze_vision_layers}/{total} blocks")

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

    # Unfreeze the logit scale
    if hasattr(clip_model, 'logit_scale'):
        clip_model.logit_scale.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in clip_model.parameters())
    print(f"  CLIP trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(clip_model, loader, optimizer, device, epoch, grad_accum=2):
    clip_model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, texts) in enumerate(loader):
        images = images.to(device)
        texts = texts.to(device)

        image_features = clip_model.encode_image(images).float()
        text_features = clip_model.encode_text(texts).float()

        logit_scale = clip_model.logit_scale.exp()
        loss = clip_contrastive_loss(image_features, text_features, logit_scale) / grad_accum

        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        if batch_idx % 100 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)}  "
                  f"loss={loss.item()*grad_accum:.4f}  logit_scale={logit_scale.item():.2f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation — image-text retrieval accuracy on a held-out set
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_retrieval(clip_model, loader, device, ks=(1, 5, 10)):
    clip_model.eval()
    all_img = []
    all_txt = []

    for images, texts in loader:
        img_feat = F.normalize(clip_model.encode_image(images.to(device)).float(), dim=-1)
        txt_feat = F.normalize(clip_model.encode_text(texts.to(device)).float(), dim=-1)
        all_img.append(img_feat)
        all_txt.append(txt_feat)

    all_img = torch.cat(all_img)
    all_txt = torch.cat(all_txt)

    # Image-to-text retrieval
    sim = all_img @ all_txt.t()
    labels = torch.arange(len(sim), device=device)

    results = {}
    for k in ks:
        topk = sim.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1)
        results[f"I2T_R@{k}"] = correct.float().mean().item() * 100

    # Text-to-image retrieval
    sim_t2i = sim.t()
    for k in ks:
        topk = sim_t2i.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1)
        results[f"T2I_R@{k}"] = correct.float().mean().item() * 100

    return results


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
            print(f"Early stopping: no improvement for {self.patience} epochs "
                  f"(best={self.best:.2f}, current={metric:.2f})")
            return True
        return False


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def save_checkpoint(path, epoch, clip_model, optimizer, scheduler, history, best_metric):
    torch.save({
        "epoch": epoch,
        "clip_state_dict": clip_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_metric": best_metric,
    }, path)


def load_checkpoint(path, clip_model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["history"], ckpt["best_metric"]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_metrics(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["loss"] for h in history], "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Caption Fine-tune Loss")
    ax1.grid(True, alpha=0.3)

    for key in ["I2T_R@1", "I2T_R@5", "I2T_R@10"]:
        ax2.plot(epochs, [h[key] for h in history], "-o", markersize=3, label=key)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall (%)")
    ax2.set_title("Image→Text Retrieval (Val)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "caption_finetune_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Curves saved to {output_dir}/caption_finetune_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on fashion image-caption pairs")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sd205/fashioncir/datasets/facap/facap")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sd205/fashioncir/output_caption_ft")
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--unfreeze_vision", type=int, default=4)
    parser.add_argument("--unfreeze_text", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- CLIP ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device)

    print("Unfreezing CLIP layers for caption fine-tuning:")
    unfreeze_clip_layers(clip_model, args.unfreeze_vision, args.unfreeze_text)

    # --- Dataset ---
    captions_dir = os.path.join(args.data_dir, "image_captions")
    full_dataset = FashionCaptionDataset(captions_dir, args.data_dir, preprocess, tokenizer)

    if args.max_samples > 0:
        full_dataset = Subset(full_dataset, range(min(args.max_samples, len(full_dataset))))

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # --- Optimizer ---
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Resume ---
    start_epoch = 1
    history = []
    best_metric = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        start_epoch, history, best_metric = load_checkpoint(
            args.resume, clip_model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best I2T_R@10={best_metric:.2f}")

    # --- Training ---
    early_stop = EarlyStopping(patience=args.patience)

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(clip_model, train_loader, optimizer, device,
                                    epoch, args.grad_accum)
        scheduler.step()

        metrics = evaluate_retrieval(clip_model, val_loader, device)
        entry = {"epoch": epoch, "loss": avg_loss, **metrics}
        history.append(entry)

        with open(os.path.join(args.output_dir, "caption_ft_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"=== Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"I2T_R@1={metrics['I2T_R@1']:.2f}  I2T_R@5={metrics['I2T_R@5']:.2f}  "
              f"I2T_R@10={metrics['I2T_R@10']:.2f} ===")

        if metrics["I2T_R@10"] > best_metric:
            best_metric = metrics["I2T_R@10"]
            torch.save({
                "epoch": epoch,
                "clip_state_dict": clip_model.state_dict(),
                "metrics": metrics,
            }, os.path.join(args.output_dir, "best_clip.pth"))
            print(f"  -> New best CLIP saved (I2T_R@10={best_metric:.2f})")

        save_checkpoint(os.path.join(args.output_dir, "resume_ckpt.pth"),
                        epoch, clip_model, optimizer, scheduler, history, best_metric)

        if early_stop.step(metrics["I2T_R@10"]):
            break

    # --- Plot ---
    plot_metrics(history, args.output_dir)

    print(f"\nStage 1 done. Fashion-adapted CLIP saved to {args.output_dir}/best_clip.pth")

    # ===========================================================================
    # Stage 2 — Train fusion module on triplets using the caption-adapted CLIP
    # ===========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Training fusion module on triplets with caption-adapted CLIP")
    print("="*70 + "\n")

    stage2_dir = os.path.join(args.output_dir, "triplet_results")
    os.makedirs(stage2_dir, exist_ok=True)

    # Load best caption-finetuned CLIP
    best_clip_path = os.path.join(args.output_dir, "best_clip.pth")
    best_clip_ckpt = torch.load(best_clip_path, map_location=device, weights_only=False)

    clip_model_s2, _, preprocess_s2 = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model_s2.load_state_dict(best_clip_ckpt["clip_state_dict"])
    clip_model_s2 = clip_model_s2.to(device).eval()
    for p in clip_model_s2.parameters():
        p.requires_grad = False
    print("Loaded caption-finetuned CLIP (frozen for triplet training)")

    tokenizer_s2 = open_clip.get_tokenizer(args.clip_model)

    # Triplet dataset — same 70/15/15 split
    triplet_dir = os.path.join(args.data_dir, "cir_triplets")
    json_files = sorted(f for f in os.listdir(triplet_dir) if f.endswith(".json"))
    rng_s2 = np.random.RandomState(42)

    train_sets, val_sets, test_sets = [], [], []
    print("Loading triplets with 70/15/15 split per category...")
    for jf in json_files:
        cat_ds = FACaPTripletDataset(
            os.path.join(triplet_dir, jf), args.data_dir, preprocess_s2, tokenizer_s2
        )
        n = len(cat_ds)
        indices = rng_s2.permutation(n)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_sets.append(Subset(cat_ds, indices[:n_train].tolist()))
        val_sets.append(Subset(cat_ds, indices[n_train:n_train + n_val].tolist()))
        test_sets.append(Subset(cat_ds, indices[n_train + n_val:].tolist()))

    s2_train = ConcatDataset(train_sets)
    s2_val = ConcatDataset(val_sets)
    s2_test = ConcatDataset(test_sets)

    if args.max_samples > 0:
        s2_train = Subset(s2_train, range(min(args.max_samples, len(s2_train))))
        s2_val = Subset(s2_val, range(min(args.max_samples // 5, len(s2_val))))
        s2_test = Subset(s2_test, range(min(args.max_samples // 5, len(s2_test))))

    print(f"Triplets — Train: {len(s2_train)}  |  Val: {len(s2_val)}  |  Test: {len(s2_test)}")

    s2_train_loader = DataLoader(s2_train, batch_size=64, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    s2_val_loader = DataLoader(s2_val, batch_size=64, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    s2_test_loader = DataLoader(s2_test, batch_size=64, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # Fusion model
    clip_dim = 512 if "B-32" in args.clip_model or "B-16" in args.clip_model else 768
    fusion_model = FashionFusionModule(clip_dim=clip_dim).to(device)
    s2_optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=1e-4)
    s2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(s2_optimizer, T_max=30)

    s2_history = []
    s2_best_r10 = 0.0
    s2_early_stop = EarlyStopping(patience=5)

    for epoch in range(1, 31):
        fusion_model.train()
        total_loss = 0.0
        for batch_idx, (ref_img, text, tar_img, _) in enumerate(s2_train_loader):
            ref_img = ref_img.to(device)
            text = text.to(device)
            tar_img = tar_img.to(device)

            with torch.no_grad():
                ref_feat = clip_model_s2.encode_image(ref_img).float()
                text_feat = clip_model_s2.encode_text(text).float()
                target_feat = F.normalize(clip_model_s2.encode_image(tar_img).float(), dim=-1)

            predicted = fusion_model(ref_feat, text_feat)
            logits = fusion_model.logit_scale.exp() * (predicted @ target_feat.t())
            labels = torch.arange(len(logits), device=device)
            loss = F.cross_entropy(logits, labels)

            s2_optimizer.zero_grad()
            loss.backward()
            s2_optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  [S2 Epoch {epoch}] Batch {batch_idx}/{len(s2_train_loader)}  loss={loss.item():.4f}")

        s2_scheduler.step()
        avg_loss = total_loss / len(s2_train_loader)

        # Eval
        metrics = evaluate_triplets(fusion_model, clip_model_s2, s2_val_loader, device)
        entry = {"epoch": epoch, "loss": avg_loss, **metrics}
        s2_history.append(entry)

        with open(os.path.join(stage2_dir, "metrics.json"), "w") as f:
            json.dump(s2_history, f, indent=2)

        print(f"=== S2 Epoch {epoch}/30  loss={avg_loss:.4f}  "
              f"R@1={metrics['R@1']:.2f}  R@5={metrics['R@5']:.2f}  R@10={metrics['R@10']:.2f} ===")

        if metrics["R@10"] > s2_best_r10:
            s2_best_r10 = metrics["R@10"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": fusion_model.state_dict(),
                "clip_checkpoint": best_clip_path,
                "metrics": metrics,
            }, os.path.join(stage2_dir, "best_model.pth"))
            print(f"  -> New best (R@10={s2_best_r10:.2f})")

        if s2_early_stop.step(metrics["R@10"]):
            break

    # Plot stage 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [h["epoch"] for h in s2_history]
    ax1.plot(epochs, [h["loss"] for h in s2_history], "b-o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Stage 2 Loss"); ax1.grid(True, alpha=0.3)
    for key in ["R@1", "R@5", "R@10"]:
        ax2.plot(epochs, [h[key] for h in s2_history], "-o", markersize=3, label=key)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Recall (%)"); ax2.set_title("Stage 2 R@K (Val)"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(stage2_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Final test eval
    print("\n=== Stage 2: Final TEST evaluation ===")
    best_s2 = torch.load(os.path.join(stage2_dir, "best_model.pth"), weights_only=False)
    fusion_model.load_state_dict(best_s2["model_state_dict"])
    test_metrics = evaluate_triplets(fusion_model, clip_model_s2, s2_test_loader, device)
    print(f"TEST — R@1={test_metrics['R@1']:.2f}  R@5={test_metrics['R@5']:.2f}  R@10={test_metrics['R@10']:.2f}")

    with open(os.path.join(stage2_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Qualitative results
    save_qualitative_results(fusion_model, clip_model_s2, s2_test, device, stage2_dir,
                             num_queries=20, top_k=5)

    print(f"\nAll done. Results in {stage2_dir}/")


if __name__ == "__main__":
    main()

# train_oct_c8.py — Train ResNet18 for OCT (8 classes)
# Classes (fixed order to match the app): ["NORMAL","DME","CNV","DRUSEN","AMD","CSR","DR","MH"]
# Works with train/val layout or flat folders. No AMP (stable on macOS/MPS).
# Strong augs, optional backbone freeze, early stopping. Saves best weights + confmat + classes.json

import argparse, json
from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 8-class order (app + trainer must match exactly) ====
APP_CLASSES = ["NORMAL","DME","CNV","DRUSEN","AMD","CSR","DR","MH"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def parse_args():
    p = argparse.ArgumentParser("Train ResNet18 for OCT (8 classes)")
    p.add_argument("--data", required=True, help="Dataset root (contains train/ val/ or flat class folders).")
    p.add_argument("--out", default="models/oct8", help="Output dir")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.15, help="Used only if no val/ folder")
    p.add_argument("--workers", type=int, default=0)  # macOS/MPS -> 0
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default="oct_resnet18_c8.pth")
    p.add_argument("--use-class-weights", action="store_true", help="Compute class weights from train set")
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze all but layer4+fc for stability")
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="Force device if needed")
    return p.parse_args()

# ---------- utils ----------
def set_seed(seed:int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(force:str):
    if force != "auto": return torch.device(force)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def safe_load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

# ---------- transforms ----------
class GrayToRGB:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            g = img.convert("L")
            img = Image.merge("RGB", (g,g,g))
        return img

def build_transforms():
    train_tf = transforms.Compose([
        GrayToRGB(),
        transforms.RandomResizedCrop(256, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.20, 0.20, 0.20, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        GrayToRGB(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

# ---------- canonicalization ----------
# Map common folder names to our APP_CLASSES exactly.
def canon(name:str) -> str|None:
    n = name.strip().lower()
    if n in ["normal","norm","normal_oct"]: return "NORMAL"
    if "dme" in n: return "DME"
    if "cnv" in n: return "CNV"
    if "drusen" in n or "drus" in n: return "DRUSEN"
    if n == "amd" or "age" in n: return "AMD"
    if n == "csr": return "CSR"
    if n == "dr" or "diabetic_retinopathy" in n: return "DR"
    if n == "mh" or "macular_hole" in n: return "MH"
    return None

def is_image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def filter_indices(imgfolder: datasets.ImageFolder) -> list[int]:
    keep = []
    dropped = defaultdict(int)
    for i, (fp, _) in enumerate(imgfolder.samples):
        cls_name = Path(fp).parent.name
        c = canon(cls_name)
        if c in APP_CLASSES and is_image_ok(Path(fp)):
            keep.append(i)
        else:
            dropped[cls_name] += 1
    if dropped:
        print("[INFO] Ignoring non-mapped/invalid files:", dict(dropped))
    return keep

def remap_imagefolder(imgfolder: datasets.ImageFolder):
    # force class_to_idx to our 8-class order
    mapping = {}
    for c in imgfolder.classes:
        cc = canon(c)
        if cc in APP_CLASSES:
            mapping[c] = APP_CLASSES.index(cc)
    imgfolder.class_to_idx = mapping
    imgfolder.classes = APP_CLASSES
    return imgfolder

def create_datasets(root: Path, val_split=0.15):
    train_tf, val_tf = build_transforms()
    tr_dir, va_dir = root / "train", root / "val"
    has_split = tr_dir.exists() and va_dir.exists()

    if has_split:
        tr_raw = datasets.ImageFolder(tr_dir, transform=train_tf)
        va_raw = datasets.ImageFolder(va_dir, transform=val_tf)
        tr_idx = filter_indices(tr_raw)
        va_idx = filter_indices(va_raw)
        if not tr_idx or not va_idx:
            raise ValueError("No valid images mapped to the 8 classes in train/val.")
        tr = Subset(remap_imagefolder(tr_raw), tr_idx)
        va = Subset(remap_imagefolder(va_raw), va_idx)
        return tr, va

    # flat layout
    full_raw = datasets.ImageFolder(root, transform=None)
    idx_keep = filter_indices(full_raw)
    if len(idx_keep) == 0:
        raise ValueError("No images mapped to the 8 classes found.")
    y = []
    for i in idx_keep:
        fp, _ = full_raw.samples[i]
        y.append(APP_CLASSES.index(canon(Path(fp).parent.name)))
    tr_idx, va_idx = train_test_split(idx_keep, test_size=val_split, random_state=42, stratify=y)
    tr = Subset(remap_imagefolder(datasets.ImageFolder(root, transform=train_tf)), tr_idx)
    va = Subset(remap_imagefolder(datasets.ImageFolder(root, transform=val_tf)), va_idx)
    return tr, va

def count_subset_labels(subset: Subset) -> Counter:
    base = subset.dataset
    counts = Counter()
    for i in subset.indices:
        fp, _ = base.samples[i]
        counts[canon(Path(fp).parent.name)] += 1
    return counts

# ---------- model / train ----------
def build_model(num_classes=8, freeze_backbone=False):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for name, p in m.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def compute_class_weights(counter: Counter) -> torch.Tensor:
    total = sum(counter.values())
    w = []
    for c in APP_CLASSES:
        freq = counter.get(c, 0)
        w.append(0.0 if freq == 0 else total / (len(APP_CLASSES) * freq))
    return torch.tensor(w, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    import numpy as _np
    return loss_sum/total, correct/total, _np.concatenate(y_true), _np.concatenate(y_pred)

def plot_confmat(y_true, y_pred, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(APP_CLASSES))))
    plt.figure(figsize=(9.5,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=APP_CLASSES, yticklabels=APP_CLASSES)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.title("OCT Confusion Matrix (8-class)")
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_png, dpi=200); plt.close()

# ---------- main ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    root = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    weights_path = out / args.weights
    classes_json = out / "oct_resnet18_c8.classes.json"
    conf_png = out / "confusion_matrix_oct_c8.png"

    # data
    train_ds, val_ds = create_datasets(root, args.val_split)
    tr_counts = count_subset_labels(train_ds)
    va_counts = count_subset_labels(val_ds)
    print("[INFO] Train counts:", tr_counts)
    print("[INFO] Val counts  :", va_counts)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=max(32, args.batch), shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    model = build_model(num_classes=len(APP_CLASSES), freeze_backbone=args.freeze_backbone).to(device)

    weights = None
    if args.use_class_weights:
        weights = compute_class_weights(tr_counts).to(device)
        print("[INFO] Using class weights:", weights.detach().cpu().numpy())
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=weights)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(12, args.epochs))

    best_acc, bad_epochs, patience = 0.0, 0, 8
    print(f"[INFO] Training for {args.epochs} epochs…")
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:5.2f}% | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(model.state_dict(), weights_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}. Best val acc: {best_acc*100:.2f}%")
                break

    if weights_path.exists():
        model.load_state_dict(safe_load_state_dict(weights_path))
    va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    print(f"[BEST] Val acc: {va_acc*100:.2f}%")

    labels = list(range(len(APP_CLASSES)))
    print("\nClassification report (val):")
    print(classification_report(y_true, y_pred, labels=labels, target_names=APP_CLASSES, digits=4))

    plot_confmat(y_true, y_pred, conf_png)
    with open(classes_json, "w") as f:
        json.dump({"classes": APP_CLASSES}, f, indent=2)

    print(f"[SAVED] Weights → {weights_path}")
    print(f"[SAVED] Confusion matrix → {conf_png}")
    print(f"[SAVED] Classes → {classes_json}")

if __name__ == "__main__":
    from sklearn.metrics import classification_report  # placed here to avoid circular import on some envs
    main()

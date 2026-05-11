# -*- coding: utf-8 -*-
"""step2_attack.py
Original file is located at
    https://colab.research.google.com/drive/1Sj-JY856nPrHuXc4rheacDQ5yQzcALj8

# Homework 3 – Step 2: Attack Face De-Identification via CNN
Reference: PETS 2016 §5.5 / arXiv:1609.00408

### Pipeline
1. Load de-identified variants from Step 1 `dataset/` folder
2. Build a lightweight CNN classifier (3 conv blocks + FC head)
3. Train one model per image variant (original, pixelized, blurred)
4. Log Top-1 & Top-5 accuracy per epoch
5. Save model weights and accuracy summary table

Hyper-parameters (paper Sub-Section 5.5): lr=0.01, momentum=0.9, weight_decay=5e-4, LR×0.5 every 25 epochs, 100 epochs, batch=16

## 1. Dataset & DataLoaders
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import copy, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


class ATTFaceDataset(Dataset):
    """
    Loads grayscale face images from:
        root/s1/1.png … root/s40/10.png
    Label = subject index (0-based).
    """
    def __init__(self, root: Path, img_size: int = 64):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir(): continue
            try:    label = int(subject_dir.name.lstrip("s")) - 1
            except: continue
            for p in sorted(subject_dir.glob("*.png")):
                self.samples.append((p, label))
        if not self.samples:
            raise FileNotFoundError(f"No images under {root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


def make_loaders(root, img_size=64, train_ratio=0.8, batch_size=16, seed=42):
    ds       = ATTFaceDataset(root, img_size)
    n_cls    = len({lbl for _, lbl in ds.samples})
    n_train  = int(len(ds) * train_ratio)
    n_test   = len(ds) - n_train
    g        = torch.Generator().manual_seed(seed)
    tr, te   = random_split(ds, [n_train, n_test], generator=g)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0),
            n_cls)

"""## 2. CNN Architecture"""
class FaceCNN(nn.Module):
    """
    Lightweight 3-conv-block CNN for grayscale 64×64 images.
    Conv(1→32) → Conv(32→64) → Conv(64→128) → FC(512) → FC(n_classes)
    """
    def __init__(self, n_classes=40, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(True), nn.MaxPool2d(2),
        )
        feat = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Device ────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

"""## 3. Training Utilities"""

def topk_acc(output, target, k=5):
    k_eff = min(k, output.size(1))
    _, pred = output.topk(k_eff, 1, True, True)
    return pred.eq(target.view(-1,1).expand_as(pred)).any(1).float().mean().item()*100


def train_epoch(model, loader, crit, opt, dev):
    model.train()
    ls, t1, t5, n = 0., 0., 0., 0
    for imgs, lbl in loader:
        imgs, lbl = imgs.to(dev), lbl.to(dev)
        opt.zero_grad(); out = model(imgs); loss = crit(out, lbl)
        loss.backward(); opt.step()
        bs=lbl.size(0); ls+=loss.item()*bs
        t1+=topk_acc(out,lbl,1)*bs; t5+=topk_acc(out,lbl,5)*bs; n+=bs
    return ls/n, t1/n, t5/n


@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval()
    ls, t1, t5, n = 0., 0., 0., 0
    for imgs, lbl in loader:
        imgs, lbl = imgs.to(dev), lbl.to(dev)
        out = model(imgs); loss = crit(out, lbl)
        bs=lbl.size(0); ls+=loss.item()*bs
        t1+=topk_acc(out,lbl,1)*bs; t5+=topk_acc(out,lbl,5)*bs; n+=bs
    return ls/n, t1/n, t5/n

"""## 4. Train One Model per Variant"""
def train_variant(variant_name, root, models_dir, figures_dir,
                   epochs=100, batch_size=16, lr=0.01,
                   momentum=0.9, weight_decay=5e-4,
                   lr_step=25, lr_gamma=0.5, img_size=64):

    print(f"\n{'='*60}\n  Variant: {variant_name}\n{'='*60}")
    tr_loader, te_loader, n_cls = make_loaders(root, img_size, batch_size=batch_size)
    print(f"  Classes:{n_cls}  Train:{len(tr_loader.dataset)}  Test:{len(te_loader.dataset)}")

    model  = FaceCNN(n_cls, img_size).to(device)
    crit   = nn.CrossEntropyLoss()
    opt    = optim.SGD(model.parameters(), lr=lr,
                       momentum=momentum, weight_decay=weight_decay)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    hist = {k:[] for k in ["tr_l","tr_t1","tr_t5","te_l","te_t1","te_t5"]}
    best_t1, best_state = 0., None

    for ep in range(1, epochs+1):
        trl,trt1,trt5 = train_epoch(model, tr_loader, crit, opt, device)
        tel,tet1,tet5 = eval_epoch (model, te_loader, crit, device)
        sched.step()
        for k,v in zip(hist, [trl,trt1,trt5,tel,tet1,tet5]): hist[k].append(v)
        if tet1 > best_t1: best_t1=tet1; best_state=copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{epochs} "
                  f"train Top1={trt1:5.1f}% Top5={trt5:5.1f}% | "
                  f"test Top1={tet1:5.1f}% Top5={tet5:5.1f}%")

    # Save model weights
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, models_dir / f"{variant_name}_cnn.pth")
    with open(models_dir / f"{variant_name}_cnn_history.json","w") as f:
        json.dump(hist, f, indent=2)

    # Plot curves
    figures_dir.mkdir(parents=True, exist_ok=True)
    eps = range(1, epochs+1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(eps, hist["tr_t1"], label="Train"); ax[0].plot(eps, hist["te_t1"], label="Test")
    ax[0].set_title(f"{variant_name} – Top-1"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.4)
    ax[1].plot(eps, hist["tr_t5"], label="Train"); ax[1].plot(eps, hist["te_t5"], label="Test")
    ax[1].set_title(f"{variant_name} – Top-5"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.4)
    plt.tight_layout()
    curve_path = figures_dir / f"step2_{variant_name}_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Best test Top-1: {best_t1:.2f}%  |  curves saved: {curve_path}")

    return {"variant": variant_name,
            "final_train_top1": hist["tr_t1"][-1],
            "final_train_top5": hist["tr_t5"][-1],
            "best_test_top1":   best_t1,
            "final_test_top5":  hist["te_t5"][-1]}

"""## 5. Run Training on All Variants"""
# ── Directories ───────────────────────────────────────────
MODELS_DIR  = Path("models")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Variants (name → subfolder in dataset/) ───────────────
VARIANTS = [
    ("original",      "original"),
    ("pixelized_b2",  "pixelized_b2"),
    ("pixelized_b4",  "pixelized_b4"),
    ("pixelized_b8",  "pixelized_b8"),
    ("pixelized_b16", "pixelized_b16"),
    ("blur_k5",       "blur_k5"),
    ("blur_k15",      "blur_k15"),
    ("blur_k45",      "blur_k45"),
    ("blur_k99",      "blur_k99"),
]

# ── Training hyper-parameters (PETS 2016 §5.5) ────────────
HP = dict(epochs=100, batch_size=16, lr=0.01,
          momentum=0.9, weight_decay=5e-4,
          lr_step=25, lr_gamma=0.5, img_size=64)

results = []
for vname, subfolder in VARIANTS:
    root = DATASET_DIR / subfolder
    if not root.exists():
        print(f"WARNING: {root} not found – skipping {vname}")
        continue
    res = train_variant(vname, root, MODELS_DIR, FIGURES_DIR, **HP)
    results.append(res)

print("\nAll variants trained.")

"""## 6. Summary Table & Bar Chart"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N_CLASSES = 40
baseline  = 100.0 / N_CLASSES   # 2.50 for AT&T Top-1
baseline5 = 100.0 * 5 / N_CLASSES  # 12.50 for AT&T Top-5

# ── Build lookup: variant_name → result dict ──────────────
lookup = {r['variant']: r for r in results}

def get(vname, key):
    """Return formatted metric string, or '—' if variant was skipped."""
    if vname not in lookup:
        return '—'
    return f"{lookup[vname][key]:.2f}"

# Columns: Dataset | Baseline | Original | b=2 | b=4 | b=8 | b=16 | k=15 | k=45 | k=99
rows = [
    ['AT&T Top 1',
     f'{baseline:.2f}',
     get('original',      'best_test_top1'),
     get('pixelized_b2',  'best_test_top1'),
     get('pixelized_b4',  'best_test_top1'),
     get('pixelized_b8',  'best_test_top1'),
     get('pixelized_b16', 'best_test_top1'),
     get('blur_k15',      'best_test_top1'),
     get('blur_k45',      'best_test_top1'),
     get('blur_k99',      'best_test_top1'),
    ],
    ['AT&T Top 5',
     f'{baseline5:.2f}',
     get('original',      'final_test_top5'),
     get('pixelized_b2',  'final_test_top5'),
     get('pixelized_b4',  'final_test_top5'),
     get('pixelized_b8',  'final_test_top5'),
     get('pixelized_b16', 'final_test_top5'),
     get('blur_k15',      'final_test_top5'),
     get('blur_k45',      'final_test_top5'),
     get('blur_k99',      'final_test_top5'),
    ],
]

col_labels = [
    'Dataset', 'Base-\nline', 'Origi-\nnal',
    '2×2', '4×4', '8×8', '16×16',
    '15', '45', '99'
]
n_cols = len(col_labels)
n_rows = len(rows)

# ── Draw table ─────────────────────────────────
fig = plt.figure(figsize=(15, 2.8))

# Top group-header row (hand-drawn as text, above the table)
ax_title = fig.add_axes([0, 0.78, 1, 0.22])
ax_title.axis('off')
ax_title.text(0.5, 0.85,
    'Accuracy of CNN Re-Identification on AT&T (40 classes) – Step 2',
    ha='center', va='top', fontsize=12, fontweight='bold', transform=ax_title.transAxes)
# Group spans (approximate x positions matching column layout)
ax_title.text(0.505, 0.2, 'Pixelization Size',
    ha='center', va='center', fontsize=9, style='italic', color='#2a6e2a',
    transform=ax_title.transAxes)
ax_title.text(0.800, 0.2, 'Gaussian parameters',
    ha='center', va='center', fontsize=9, style='italic', color='#8b4000',
    transform=ax_title.transAxes)
# Underlines for group headers
for x0, x1, color in [(.36, .65, '#2a6e2a'), (.68, .93, '#8b4000')]:
    ax_title.plot([x0, x1], [0.08, 0.08], color=color, linewidth=1.2,
                  transform=ax_title.transAxes)

ax = fig.add_axes([0, 0, 1, 0.80])
ax.axis('off')

tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)

# Style header
for col in range(n_cols):
    cell = tbl[0, col]
    cell.set_facecolor('#e8e8e8')
    cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('#555')

# Style data rows
for row in range(1, n_rows + 1):
    for col in range(n_cols):
        cell = tbl[row, col]
        cell.set_edgecolor('#aaa')
        if col == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f5f5f5')
        elif col == 1:
            cell.set_facecolor('#fafafa')   # baseline
        elif col == 2:
            cell.set_facecolor('#fafafa')   # original
        elif 3 <= col <= 6:
            cell.set_facecolor('#f0fff0')   # pixelization
        else:
            cell.set_facecolor('#fff5ee')   # gaussian

plt.tight_layout()
table_path = FIGURES_DIR / 'step2_summary_table.png'
plt.savefig(table_path, dpi=180, bbox_inches='tight')
plt.show()
print(f'Saved table figure: {table_path}')

# ── Plain-text version ────────────────────────────────────
SEP = '-' * 92
print('\n' + SEP)
print(f'{"":22} {"Base-":>7} {"Origi-":>7}  {"── Pixelization Size ────────":^28}  {"── Gaussian k ──":^20}')
print(f'{"Dataset":<22} {"line":>7} {"nal":>7}  {"2×2":>6} {"4×4":>6} {"8×8":>6} {"16×16":>6}  {"15":>6} {"45":>6} {"99":>6}')
print(SEP)
for row in rows:
    vals = '  '.join(f'{v:>6}' for v in row[1:])
    print(f'{row[0]:<22} {vals}')
print(SEP)
print(f'Baseline Top-1 = {baseline:.2f}%  (1/{N_CLASSES})   |   Baseline Top-5 = {baseline5:.2f}%  (5/{N_CLASSES})')

# ── Save JSON for Step 3 ──────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
with open(MODELS_DIR / 'step2_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nStep 2 complete.  Models → {MODELS_DIR}  |  Figures → {FIGURES_DIR}')

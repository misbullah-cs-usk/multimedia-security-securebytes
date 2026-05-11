"""
Homework 3 — Step 3: Differential Privacy Defense
=================================================

This script:
1. Loads strongest non-private variants from Step 1
      - pixelized_b16
      - blur_k99

2. Applies Differential Privacy noise
      - Laplace mechanism
      - epsilons = [0.1, 0.5, 1.0]

3. Saves DP-obfuscated datasets

4. Loads Step 2 trained CNN models

5. Runs inference-only attack evaluation
      - NO retraining

6. Computes:
      - Top-1 accuracy
      - Top-5 accuracy
      - MSE
      - SSIM

7. Produces:
      - CSV results
      - JSON results
      - Figures

Compatible with your existing directory structure.

------------------------------------------------------------
Expected folders:
------------------------------------------------------------

dataset/
    original/
    pixelized_b16/
    blur_k99/

models/
    pixelized_b16_cnn.pth
    blur_k99_cnn.pth

------------------------------------------------------------
Run:
------------------------------------------------------------

python step3_dp_defense.py

"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json
import copy

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image

from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")

DP_DIR = Path("dataset_dp")
FIGURES_DIR = Path("figures_step3")
RESULTS_DIR = Path("results_step3")

DP_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPSILONS = [0.1, 0.5, 1.0]
#EPSILONS = [1, 5, 10, 20, 50, 100]

IMG_SIZE = 64
BATCH_SIZE = 16
N_CLASSES = 40

SENSITIVITY = 10.0

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Dataset Loader (same as Step 2)
# ============================================================

class ATTFaceDataset(Dataset):

    def __init__(self, root: Path, img_size: int = 64):

        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for subject_dir in sorted(root.iterdir()):

            if not subject_dir.is_dir():
                continue

            try:
                label = int(subject_dir.name.lstrip("s")) - 1
            except:
                continue

            for p in sorted(subject_dir.glob("*.png")):
                self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"No images under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("L")

        return self.transform(img), label


def make_loaders(
    root,
    img_size=64,
    train_ratio=0.8,
    batch_size=16,
    seed=42
):

    ds = ATTFaceDataset(root, img_size)

    n_train = int(len(ds) * train_ratio)
    n_test = len(ds) - n_train

    g = torch.Generator().manual_seed(seed)

    tr, te = random_split(ds, [n_train, n_test], generator=g)

    train_loader = DataLoader(
        tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


# ============================================================
# CNN Model (same as Step 2)
# ============================================================

class FaceCNN(nn.Module):

    def __init__(self, n_classes=40, img_size=64):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        feat = (img_size // 8) ** 2 * 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# Metrics
# ============================================================

def topk_acc(output, target, k=5):

    k_eff = min(k, output.size(1))

    _, pred = output.topk(k_eff, 1, True, True)

    correct = pred.eq(
        target.view(-1, 1).expand_as(pred)
    )

    return correct.any(1).float().mean().item() * 100


# ============================================================
# Differential Privacy Noise
# ============================================================

def add_laplace_noise(
    image,
    epsilon,
    sensitivity=255.0
):
    """
    Laplace Mechanism
    """
    
    scale = sensitivity / epsilon

    noise = np.random.laplace(
        loc=0.0,
        scale=scale,
        size=image.shape
    )

    # Add noise
    noisy = image.astype(np.float32) + noise

    # Clip back to valid range
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

# ============================================================
# Create DP Dataset
# ============================================================

def create_dp_dataset(
    input_root,
    output_root,
    epsilon
):

    print(f"\nCreating DP dataset:")
    print(f"  Input : {input_root}")
    print(f"  Output: {output_root}")
    print(f"  epsilon = {epsilon}")

    output_root.mkdir(parents=True, exist_ok=True)

    total = 0

    for subject_dir in sorted(input_root.iterdir()):

        if not subject_dir.is_dir():
            continue

        out_subject = output_root / subject_dir.name
        out_subject.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(subject_dir.glob("*.png")):

            img = cv2.imread(
                str(img_path),
                cv2.IMREAD_GRAYSCALE
            )

            dp_img = add_laplace_noise(
                img,
                epsilon=epsilon,
                sensitivity=SENSITIVITY
            )

            save_path = out_subject / img_path.name

            cv2.imwrite(str(save_path), dp_img)

            total += 1

    print(f"  Saved {total} DP images")


# ============================================================
# MSE + SSIM
# ============================================================

def compute_mse_ssim(
    original_root,
    dp_root
):

    mses = []
    ssims = []

    for subject_dir in sorted(original_root.iterdir()):

        if not subject_dir.is_dir():
            continue

        dp_subject = dp_root / subject_dir.name

        for img_path in sorted(subject_dir.glob("*.png")):

            dp_path = dp_subject / img_path.name

            orig = cv2.imread(
                str(img_path),
                cv2.IMREAD_GRAYSCALE
            )

            dp = cv2.imread(
                str(dp_path),
                cv2.IMREAD_GRAYSCALE
            )

            mse = np.mean(
                (orig.astype(np.float32) -
                 dp.astype(np.float32)) ** 2
            )

            ssim_score = ssim(
                orig,
                dp,
                data_range=255
            )

            mses.append(mse)
            ssims.append(ssim_score)

    return float(np.mean(mses)), float(np.mean(ssims))


# ============================================================
# Evaluate Saved Model
# ============================================================

@torch.no_grad()
def evaluate_saved_model(
    model_path,
    dataset_root,
    n_classes=40,
    img_size=64,
    batch_size=16
):

    print(f"\nEvaluating model:")
    print(f"  Model  : {model_path.name}")
    print(f"  Dataset: {dataset_root.name}")

    _, test_loader = make_loaders(
        dataset_root,
        img_size=img_size,
        batch_size=batch_size
    )

    model = FaceCNN(
        n_classes=n_classes,
        img_size=img_size
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    top1_total = 0
    top5_total = 0
    n = 0

    for imgs, labels in test_loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        bs = labels.size(0)

        top1_total += topk_acc(outputs, labels, 1) * bs
        top5_total += topk_acc(outputs, labels, 5) * bs

        n += bs

    top1 = top1_total / n
    top5 = top5_total / n

    print(f"  Top-1 = {top1:.2f}%")
    print(f"  Top-5 = {top5:.2f}%")

    return top1, top5


# ============================================================
# Main
# ============================================================

def main():

    print("\n====================================================")
    print("STEP 3 — DIFFERENTIAL PRIVACY DEFENSE")
    print("====================================================")

    # --------------------------------------------------------
    # Create DP datasets
    # --------------------------------------------------------

    base_variants = [
        ("pixelized_b16", DATASET_DIR / "pixelized_b16"),
        ("blur_k99", DATASET_DIR / "blur_k99"),
    ]

    for variant_name, input_root in base_variants:

        for eps in EPSILONS:

            output_root = DP_DIR / f"{variant_name}_eps{eps}"

            create_dp_dataset(
                input_root=input_root,
                output_root=output_root,
                epsilon=eps
            )

    # --------------------------------------------------------
    # Evaluate DP datasets
    # --------------------------------------------------------

    results = []

    # Evaluate Step 2 non-DP models on their matching datasets
    step2_variants = {
        "original": DATASET_DIR / "original",
        "pixelized_b2": DATASET_DIR / "pixelized_b2",
        "pixelized_b4": DATASET_DIR / "pixelized_b4",
        "pixelized_b8": DATASET_DIR / "pixelized_b8",
        "pixelized_b16": DATASET_DIR / "pixelized_b16",
        "blur_k5": DATASET_DIR / "blur_k5",
        "blur_k15": DATASET_DIR / "blur_k15",
        "blur_k45": DATASET_DIR / "blur_k45",
        "blur_k99": DATASET_DIR / "blur_k99",
    }
    
    print("\n====================================================")
    print("EVALUATING STEP 2 NON-DP BASELINES")
    print("====================================================")
    
    for variant_name, dataset_root in step2_variants.items():
    
        model_path = MODELS_DIR / f"{variant_name}_cnn.pth"
    
        if not dataset_root.exists():
            print(f"Skipping missing dataset: {dataset_root}")
            continue
    
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue
    
        top1, top5 = evaluate_saved_model(
            model_path=model_path,
            dataset_root=dataset_root,
            n_classes=N_CLASSES,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    
        mse, ssim_score = compute_mse_ssim(
            DATASET_DIR / "original",
            dataset_root
        )
    
        results.append({
            "variant": variant_name,
            "epsilon": "NP",
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "baseline_top1": 100 / N_CLASSES,
            "baseline_top5": 100 * 5 / N_CLASSES,
            "mse": mse,
            "ssim": ssim_score,
        })
    
    
    # Only these two trained models are reused for DP inference
    model_lookup = {
        "pixelized_b16": MODELS_DIR / "pixelized_b16_cnn.pth",
        "blur_k99": MODELS_DIR / "blur_k99_cnn.pth",
    }

    for variant_name in model_lookup.keys():

        model_path = model_lookup[variant_name]

        for eps in EPSILONS:

            dp_dataset_root = DP_DIR / f"{variant_name}_eps{eps}"

            # --------------------------------------------
            # CNN attack evaluation
            # --------------------------------------------

            top1, top5 = evaluate_saved_model(
                model_path=model_path,
                dataset_root=dp_dataset_root,
                n_classes=N_CLASSES,
                img_size=IMG_SIZE,
                batch_size=BATCH_SIZE
            )

            # --------------------------------------------
            # Utility metrics
            # --------------------------------------------

            mse, ssim_score = compute_mse_ssim(
                DATASET_DIR / "original",
                dp_dataset_root
            )

            # --------------------------------------------
            # Save result
            # --------------------------------------------

            results.append({

                "variant": variant_name,
                "epsilon": eps,

                "top1_accuracy": top1,
                "top5_accuracy": top5,

                "baseline_top1": 100 / N_CLASSES,
                "baseline_top5": 100 * 5 / N_CLASSES,

                "mse": mse,
                "ssim": ssim_score,
            })

    # ========================================================
    # Save Results
    # ========================================================

    df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "step3_dp_results.csv"
    json_path = RESULTS_DIR / "step3_dp_results.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")

    # ========================================================
    # Print Table
    # ========================================================

    print("\n====================================================")
    print("FINAL RESULTS")
    print("====================================================")

    print(df)

    # ========================================================
    # Plot: MSE vs epsilon
    # ========================================================

    df_dp = df[df["epsilon"] != "NP"].copy()
    df_dp["epsilon"] = df_dp["epsilon"].astype(float)

    plt.figure(figsize=(6, 4))

    for variant in df_dp["variant"].unique():

        sub = df_dp[df_dp["variant"] == variant]

        plt.plot(
            sub["epsilon"],
            sub["mse"],
            marker="o",
            label=variant
        )

    plt.xlabel("epsilon")
    plt.ylabel("MSE")
    plt.title("MSE vs epsilon")
    plt.grid(alpha=0.3)
    plt.legend()

    mse_fig = FIGURES_DIR / "step3_mse_vs_epsilon.png"

    plt.savefig(
        mse_fig,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {mse_fig}")

    # ========================================================
    # Plot: SSIM vs epsilon
    # ========================================================

    plt.figure(figsize=(6, 4))

    for variant in df_dp["variant"].unique():

        sub = df_dp[df_dp["variant"] == variant]

        plt.plot(
            sub["epsilon"],
            sub["ssim"],
            marker="o",
            label=variant
        )

    plt.xlabel("epsilon")
    plt.ylabel("SSIM")
    plt.title("SSIM vs epsilon")
    plt.grid(alpha=0.3)
    plt.legend()

    ssim_fig = FIGURES_DIR / "step3_ssim_vs_epsilon.png"

    plt.savefig(
        ssim_fig,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {ssim_fig}")

    # ========================================================
    # Plot: Attack accuracy vs epsilon
    # ========================================================

    plt.figure(figsize=(6, 4))

    for variant in df_dp["variant"].unique():

        sub = df_dp[df_dp["variant"] == variant]

        plt.plot(
            sub["epsilon"],
            sub["top1_accuracy"],
            marker="o",
            label=f"{variant}"
        )

    plt.axhline(
        100 / N_CLASSES,
        linestyle="--",
        label="Baseline"
    )

    plt.xlabel("epsilon")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Attack Accuracy vs epsilon")
    plt.grid(alpha=0.3)
    plt.legend()

    acc_fig = FIGURES_DIR / "step3_attack_accuracy_vs_epsilon.png"

    plt.savefig(
        acc_fig,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {acc_fig}")

    # ========================================================
    # Sample visualization
    # ========================================================

    sample_orig = cv2.imread(
        str(DATASET_DIR / "original" / "s1" / "1.png"),
        cv2.IMREAD_GRAYSCALE
    )

    fig, axes = plt.subplots(
        2,
        len(EPSILONS) + 1,
        figsize=(12, 5)
    )

    methods = [
        ("pixelized_b16", 0),
        ("blur_k99", 1)
    ]

    for row, (variant, _) in enumerate(methods):

        axes[row, 0].imshow(sample_orig, cmap="gray")
        axes[row, 0].set_title("Original")
        axes[row, 0].axis("off")

        for col, eps in enumerate(EPSILONS, start=1):

            img_path = (
                DP_DIR /
                f"{variant}_eps{eps}" /
                "s1" /
                "1.png"
            )

            img = cv2.imread(
                str(img_path),
                cv2.IMREAD_GRAYSCALE
            )

            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(f"{variant}\nε={eps}")
            axes[row, col].axis("off")

    plt.tight_layout()

    sample_fig = FIGURES_DIR / "step3_dp_examples.png"

    plt.savefig(
        sample_fig,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {sample_fig}")

    # ========================================================
    # Visual Comparison Figure
    # ========================================================
    
    print("\nGenerating comparison figure...")
    
    sample_subjects = ["s1", "s5", "s10", "s15", "s20"]
    sample_image = "1.png"
    
    # Select epsilon for comparison examples
    VIS_EPS = 1.0
    
    fig, axes = plt.subplots(
        len(sample_subjects),
        5,
        figsize=(14, 12)
    )
    
    col_titles = [
        "Original",
        "NP Pixelized",
        f"DP Pixelized\nε={VIS_EPS}",
        "NP Blur",
        f"DP Blur\nε={VIS_EPS}"
    ]
    
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)
    
    for row, subject in enumerate(sample_subjects):
    
        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------
    
        orig_path = (
            DATASET_DIR /
            "original" /
            subject /
            sample_image
        )
    
        np_pixel_path = (
            DATASET_DIR /
            "pixelized_b16" /
            subject /
            sample_image
        )
    
        dp_pixel_path = (
            DP_DIR /
            f"pixelized_b16_eps{VIS_EPS}" /
            subject /
            sample_image
        )
    
        np_blur_path = (
            DATASET_DIR /
            "blur_k99" /
            subject /
            sample_image
        )
    
        dp_blur_path = (
            DP_DIR /
            f"blur_k99_eps{VIS_EPS}" /
            subject /
            sample_image
        )
    
        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------
    
        orig = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
        np_pixel = cv2.imread(str(np_pixel_path), cv2.IMREAD_GRAYSCALE)
        dp_pixel = cv2.imread(str(dp_pixel_path), cv2.IMREAD_GRAYSCALE)
        np_blur = cv2.imread(str(np_blur_path), cv2.IMREAD_GRAYSCALE)
        dp_blur = cv2.imread(str(dp_blur_path), cv2.IMREAD_GRAYSCALE)
    
        imgs = [
            orig,
            np_pixel,
            dp_pixel,
            np_blur,
            dp_blur
        ]
    
        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
    
        for col, img in enumerate(imgs):
    
            axes[row, col].imshow(
                img,
                cmap="gray",
                vmin=0,
                vmax=255
            )
    
            axes[row, col].axis("off")
    
            if col == 0:
                axes[row, col].set_ylabel(
                    subject,
                    fontsize=11,
                    rotation=90
                )
    
    plt.tight_layout()
    
    comparison_fig = (
        FIGURES_DIR /
        "step3_visual_comparison.png"
    )
    
    plt.savefig(
        comparison_fig,
        dpi=180,
        bbox_inches="tight"
    )
    
    plt.close()
    
    print(f"Saved comparison figure: {comparison_fig}")

    print("\n====================================================")
    print("STEP 3 COMPLETE")
    print("====================================================")

    print(f"DP datasets : {DP_DIR.resolve()}")
    print(f"Figures     : {FIGURES_DIR.resolve()}")
    print(f"Results     : {RESULTS_DIR.resolve()}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()

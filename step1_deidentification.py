"""
Homework 3 - Step 1: Face De-Identification
Methods: Gaussian Blurring and Pixelization
Dataset: AT&T ORL Faces (40 subjects, 10 images each = 400 total)

Pipeline:
  1. Load AT&T faces via sklearn (fetches the ORL dataset automatically)
  2. Detect face ROI with OpenCV Haar Cascade (falls back to full image for
     pre-cropped datasets like AT&T)
  3. Apply Gaussian blur at k = 15, 45, 99
  4. Apply pixelization at block size b = 4, 8, 16
  5. Save comparison figures and de-identified image folders
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces


# ──────────────────────────────────────────────
# 1. Dataset Loading
# ──────────────────────────────────────────────

def load_att_faces():
    """
    Fetch the AT&T ORL Faces dataset via sklearn.
    Returns:
        images  – (400, 64, 64) uint8 grayscale array
        targets – (400,) int array of subject IDs 0-39
    """
    print("Downloading AT&T ORL Faces dataset (cached after first run)...")
    data = fetch_olivetti_faces(shuffle=False)
    # sklearn returns float64 in [0,1]; convert to uint8 for OpenCV
    images_u8 = (data.images * 255).astype(np.uint8)
    return images_u8, data.target


# ──────────────────────────────────────────────
# 2. Face Detection
# ──────────────────────────────────────────────

# Load cascade once (avoids reloading on every call)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_roi(image):
    """
    Detect the largest frontal face bounding box using Haar Cascade.
    Falls back to the full image if no face is found — correct behaviour
    for AT&T images which are already tightly cropped to the face.

    Returns (x, y, w, h) as ints.
    """
    faces = _FACE_CASCADE.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
    )
    if len(faces) > 0:
        # Pick the largest detected rectangle by area
        areas = [w * h for (x, y, w, h) in faces]
        return tuple(int(v) for v in faces[int(np.argmax(areas))])
    else:
        # Fallback: treat the entire image as the face region
        h, w = image.shape[:2]
        return 0, 0, w, h


# ──────────────────────────────────────────────
# 3. De-Identification Methods
# ──────────────────────────────────────────────

def apply_gaussian_blur(image, k):
    """
    Gaussian blur on the detected face ROI.
    k: kernel size (odd integer). Larger k → heavier smoothing → harder to recognise.
    """
    # Ensure kernel size is a positive odd number
    k = int(k)
    if k % 2 == 0:
        k += 1

    result = image.copy()
    x, y, w, h = detect_face_roi(image)
    face_roi = result[y : y + h, x : x + w]
    result[y : y + h, x : x + w] = cv2.GaussianBlur(face_roi, (k, k), 0)
    return result


def apply_pixelization(image, b):
    """
    Pixelate the detected face ROI.
    b: block size. Downsample by factor b then upsample with NEAREST interpolation.
    Larger b → coarser mosaic → stronger de-identification but lower utility.
    """
    b = int(b)
    result = image.copy()
    x, y, w, h = detect_face_roi(image)
    face_roi = result[y : y + h, x : x + w]

    # Clamp to avoid zero dimensions
    small_w = max(1, w // b)
    small_h = max(1, h // b)

    # Downsample (average over each block) then scale back up (nearest-neighbor)
    small = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    result[y : y + h, x : x + w] = pixelated
    return result


# ──────────────────────────────────────────────
# 4. Comparison Figures (matching Slide 6 layout)
# ──────────────────────────────────────────────

def _make_comparison_figure(sample_images, param_values, transform_fn,
                             col_labels, title, save_path):
    """
    Generic helper: one row per sample image, one column per parameter value
    (plus the original in column 0).
    """
    n_rows = len(sample_images)
    n_cols = 1 + len(param_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.4),
        squeeze=False,
    )

    for row, img in enumerate(sample_images):
        # Column 0 – original
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("(a) orig", fontsize=10)

        # Remaining columns – de-identified variants
        for col, (param, label) in enumerate(zip(param_values, col_labels), start=1):
            out = transform_fn(img, param)
            axes[row, col].imshow(out, cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label, fontsize=10)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {save_path}")


def create_blur_figure(sample_images, k_values, save_path):
    letters = list("efghijklmn")
    labels = [f"({letters[i]}) k={k}" for i, k in enumerate(k_values)]
    _make_comparison_figure(
        sample_images, k_values, apply_gaussian_blur,
        labels,
        "Gaussian Blur De-Identification",
        save_path,
    )


def create_pixelization_figure(sample_images, b_values, save_path):
    letters = list("bcdefghijk")
    labels = [f"({letters[i]}) b={b}" for i, b in enumerate(b_values)]
    _make_comparison_figure(
        sample_images, b_values, apply_pixelization,
        labels,
        "Pixelization De-Identification",
        save_path,
    )


# ──────────────────────────────────────────────
# 5. Save De-Identified Dataset to Disk
# ──────────────────────────────────────────────

def save_deidentified_dataset(images, targets, output_dir, k_values, b_values):
    """
    Write every de-identified image to:
        output_dir/blur_k{k}/s{subject}/{local_idx}.png
        output_dir/pixelized_b{b}/s{subject}/{local_idx}.png
        output_dir/original/s{subject}/{local_idx}.png
    """
    output_dir = Path(output_dir)
    n_subjects = int(targets.max()) + 1

    def _save_batch(method_folder, transform_fn, param, label):
        # Pre-create subject sub-folders
        for sid in range(1, n_subjects + 1):
            (method_folder / f"s{sid}").mkdir(parents=True, exist_ok=True)

        for idx, (img, subject) in enumerate(zip(images, targets)):
            local_idx = idx % 10 + 1   # images per subject are indexed 1-10
            out = transform_fn(img, param) if transform_fn else img
            path = method_folder / f"s{int(subject) + 1}" / f"{local_idx}.png"
            cv2.imwrite(str(path), out)
        print(f"  Saved {len(images)} images -> {method_folder}")

    print("\nSaving Gaussian blur variants...")
    for k in k_values:
        _save_batch(output_dir / f"blur_k{k}", apply_gaussian_blur, k, f"k={k}")

    print("Saving pixelization variants...")
    for b in b_values:
        _save_batch(output_dir / f"pixelized_b{b}", apply_pixelization, b, f"b={b}")

    print("Saving originals...")
    _save_batch(output_dir / "original", None, None, "original")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # Gaussian blur kernel sizes (must be odd)
    K_VALUES = [5, 15, 45, 99]
    # Pixelization block sizes
    B_VALUES = [2, 4, 8, 16]

    FIGURES_DIR = Path("figures")
    DATASET_DIR = Path("dataset")
    FIGURES_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────
    images, targets = load_att_faces()
    print(f"Dataset: {len(images)} images, "
          f"{len(np.unique(targets))} subjects, "
          f"image shape: {images[0].shape}")

    # ── Sample images for the comparison figures ──
    # One image per subject (first of each group of 10), pick 4 subjects
    sample_indices = [0, 10, 20, 30]
    sample_images = [images[i] for i in sample_indices]

    # ── Figures ───────────────────────────────
    print("\nGenerating comparison figures...")
    create_blur_figure(
        sample_images, K_VALUES,
        FIGURES_DIR / "step1_gaussian_blur_comparison.png",
    )
    create_pixelization_figure(
        sample_images, B_VALUES,
        FIGURES_DIR / "step1_pixelization_comparison.png",
    )

    # ── Save full de-identified dataset ───────
    save_deidentified_dataset(images, targets, DATASET_DIR, K_VALUES, B_VALUES)

    print(f"\nStep 1 complete.")
    print(f"  Figures -> {FIGURES_DIR.resolve()}")
    print(f"  Dataset -> {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    main()

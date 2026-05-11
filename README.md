# Multimedia Security - Face De-Identification & Its Attacks and Defenses

## Course Name: Data Privacy and Security (1142CS5164701)
### Group Name: SecureBytes
Member:
1. Alim Misbullah D11415803	
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

## Project Overview
In this project, we investigate privacy-preserving face de-identification using Gaussian blurring, pixelization, and Differential Privacy (DP) techniques. The experiments are conducted using the AT&T Face Dataset, where facial images are first obfuscated using traditional de-identification methods and then evaluated against CNN-based re-identification attacks. A convolutional neural network (CNN) is trained to measure how accurately identities can still be recovered from de-identified images. To further strengthen privacy protection, Differential Privacy noise is added to the strongest obfuscated variants using the Laplace mechanism with different privacy budgets (ε values). The effectiveness of the proposed defense is evaluated using Top-1 and Top-5 attack accuracy, Mean Squared Error (MSE), and Structural Similarity Index (SSIM). The project aims to analyze the tradeoff between privacy protection and image utility in multimedia data privacy systems.

## Objectives
  - To implement face de-identification techniques using Gaussian blurring and pixelization.
  - To evaluate the effectiveness of different obfuscation parameters in protecting facial identity.
  - To develop a CNN-based attack model capable of re-identifying individuals from de-identified images.
  - To analyze the impact of blur kernel size and pixelization block size on attack accuracy.
  - To apply Differential Privacy (DP) noise using the Laplace mechanism to further enhance privacy protection.
  - To evaluate the privacy-utility tradeoff using:
    - Top-1 and Top-5 accuracy
    - Mean Squared Error (MSE)
    - Structural Similarity Index (SSIM)
  - To determine whether Differential Privacy can reduce CNN re-identification performance while maintaining acceptable image quality.

## How to Run the Project

### 1. Clone Repository
```
git clone https://github.com/misbullah-cs-usk/multimedia-security-securebytes
cd multimedia-security-securebytes
```

### 2. Create Python Environment
```
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

## Project Structure
```
./
├── step1_deidentification.py
├── step2_attack.py
├── step3_dp_defense.py
├── dataset/
├── dataset_dp/
├── models/
├── figures/
├── figures_step3/
└── results_step3/
```

## Step 1 — Face De-Identification
This step:
- Downloads the AT&T ORL Face Dataset
- Applies:
  - Gaussian blur
  - Pixelization
- Saves:
  - comparison figures
  - de-identified datasets

Run
```
python3 step1_deidentification.py
```

Generated folders
```
dataset/
├── original/
├── pixelized_b2/
├── pixelized_b4/
├── pixelized_b8/
├── pixelized_b16/
├── blur_k5/
├── blur_k15/
├── blur_k45/
└── blur_k99/
```

Generated figures
```
figures/
├── step1_gaussian_blur_comparison.png
└── step1_pixelization_comparison.png
```

## Step 2 — CNN Re-Identification Attack
This step:
- Loads the datasets generated in Step 1
- Trains one CNN model for each image variant
- Evaluates:
  - Top-1 accuracy
  - Top-5 accuracy
- Saves:
  - trained models
  - training history
  - summary figures

Run
```
python3 step2_attack.py
```

Generated models:
```
models/
├── original_cnn.pth
├── pixelized_b16_cnn.pth
├── blur_k99_cnn.pth
└── ...
```

Generated figures:
```
figures/
├── step2_summary_table.png
├── step2_original_curves.png
└── ...
```

Table Result

| Dataset | Baseline | Original | 2×2 | 4×4 | 8×8 | 16×16 | k=15 | k=45 | k=99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AT&T Top-1 (%) | 2.50 | 92.50 | 96.25 | 85.00 | 87.50 | 80.00 | 91.25 | 76.25 | 62.50 |
| AT&T Top-5 (%) | 12.50 | 97.50 | 97.50 | 95.00 | 96.25 | 95.00 | 98.75 | 95.00 | 82.50 |

## Step 3 — Differential Privacy Defense
This step:
- Applies Differential Privacy noise using the Laplace mechanism
- Creates DP-protected datasets
- Re-runs CNN attack evaluation
- Computes:
  - Top-1 accuracy
  - Top-5 accuracy
  - MSE
  - SSIM
- Generates result plots and visual comparison images

Run
```
python3 step3_dp_defense.py
```

Generated DP datasets:
```
dataset_dp/
├── pixelized_b16_eps0.1/
├── pixelized_b16_eps0.5/
├── pixelized_b16_eps1.0/
├── blur_k99_eps0.1/
├── blur_k99_eps0.5/
└── blur_k99_eps1.0/
```

Generated figures:
```
figures_step3/
├── step3_mse_vs_epsilon.png
├── step3_ssim_vs_epsilon.png
├── step3_attack_accuracy_vs_epsilon.png
├── step3_dp_examples.png
└── step3_visual_comparison.png
```

Generated results:
```
results_step3/
├── step3_dp_results.csv
└── step3_dp_results.json
```

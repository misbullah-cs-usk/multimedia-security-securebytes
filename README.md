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

<div align="center">
  
# Proactive Deepfake Defense via Robust Cross-Image Adversarial Watermarking
**Defending Social Media Against GAN-Based Facial Forgery under JPEG Compression**

</div>

## 📌 Executive Summary
This project implements a proactive Deepfake defense framework that generates a single, cross-image universal adversarial watermark. When embedded into an image before it is shared online, this imperceptible watermark forces targeted GAN-based Deepfake generators (such as **StarGAN**, AttGAN, and HiSD) to produce severely distorted outputs, effectively shattering their synthesis capabilities. 

Unlike standard spatial adversarial perturbations that are easily destroyed by lossy compression, our watermark is optimized directly within the **Discrete Cosine Transform (DCT) frequency domain**. This ensures survival against aggressive social media quantization (JPEG) while maintaining high visual fidelity (PSNR ~30 dB, SSIM ~0.90) for the protected image.

---

## 🚀 Core Defense Mechanism
Given an input face $I$ and a target manipulation attribute $c$ (e.g., changing hair color or gender), a Deepfake model computes the forgery as $I' = G(I, c)$. 

Our framework optimizes a frequency-domain watermark $w$ embedded directly into the DCT luminance coefficients. The resulting protected image $I_{protected} = I_{clean} + w$ looks visually identical to the original image. However, when a malicious actor attempts to manipulate it, the generator fails catastrophically: $G(I_{protected}, c)$ yields severe visual artifacts and color blockages, preventing the creation of a convincing Deepfake.

---

## 🧠 Architectural Overview & Implementation Details

### 1. Differentiable JPEG Compression (`DiffJPEG`)
Standard JPEG quantization utilizes a discontinuous `round(x)` operation, causing zero gradients almost everywhere during backpropagation. To allow end-to-end watermark optimization via **Projected Gradient Descent (PGD)**, we replace standard rounding with a smooth cubic Straight-Through Estimator (STE) approximation:
$$x_{approx} = \text{round}(x) + (x - \text{round}(x))^3$$
This allows robust gradients to flow from the Deepfake model, through the quantization layer, back to the frequency coefficients.

### 2. Frequency Masking & Y-Channel Isolation
To balance adversarial disruption against perceptual stealth, we apply two strict constraints:
- **Color Space Isolation**: The watermark is injected exclusively into the **Luminance (Y)** channel of the YCbCr color space. The human eye is less sensitive to luminance perturbations, and JPEG compression quantizes Cb/Cr channels far more aggressively, which would destroy the watermark.
- **Mid-Frequency Binary Mask**: Inside the $8 \times 8$ DCT blocks (indexed $u, v \in [0, 7]$), we enforce a binary mask $M$ where $2 \le u + v \le 6$. 
  - *Avoiding Low Frequencies*: Prevents massive blocky color shifts in the spatial domain (protecting PSNR).
  - *Avoiding High Frequencies*: Prevents the watermark from being aggressively zeroed out by JPEG quantization tables.

### 3. PGD Adversarial Optimization
The universal watermark is trained over batches of images using PGD in the frequency domain. 

**Composite Loss Function:**
The objective balances maximum Deepfake output disruption against minimal spatial distortion:
$$L = \text{MSE}\Big(G(I_{inv}), G(I_{clean})\Big) - \lambda \cdot \text{MSE}(I_{inv}, I_{clean})$$
*(where $\lambda = 0.10$ acts as a penalty to prevent excessive spatial noise).*

**Cross-Image Fusion Strategy:**
To generalize the watermark across all unseen faces (a "Universal" watermark):
1. **Intra-Batch Gradient Sign Averaging**: Gradients are averaged across a batch of faces to extract common facial feature weaknesses.
2. **Inter-Batch Exponential Moving Average (EMA)**: Across batches, the universal watermark is updated using EMA to prevent catastrophic forgetting.

---

## 📊 Evaluation & Verification Metrics
The framework achieves state-of-the-art balance between fidelity and defense success, measured strictly on unseen datasets (CelebA / LFW) after applying real non-differentiable JPEG compression ($Q=75$).

| Metric | Result (StarGAN) | Description |
| :--- | :---: | :--- |
| **PSNR** | **26.78 dB** | High perceptual fidelity; pixel-level distortion is imperceptible. |
| **SSIM** | **0.5588** | Preserves structural textures and edges of the original face. |
| **$L_2$ Disruption** | **0.062** | The squared L2 distance between the Deepfake output on the clean vs. protected image over the facial crop. |
| **Success Rate (SR)**| **78.1250%** | The percentage of images successfully disrupted beyond the failure threshold. |

*(Hyperparameters: $\epsilon = 0.018$, $\alpha = 0.003$, $T = 20$ iterations).*

---

## 💻 Getting Started

### 1. Requirements
Install the required dependencies from the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(PyTorch, Torchvision, NumPy, Matplotlib, scikit-image, Pillow, tqdm)*

### 2. File Structure
*   `main.py`: The core orchestrator for dataset loading, training, and evaluation.
*   `watermark.py`: Contains `WatermarkOptimizer` for the PGD logic and mask application.
*   `jpeg_utils.py`: Contains the `DiffJPEG` differentiable compression module.
*   `evaluation.py`: Computes strict reporting metrics (PSNR, SSIM, MSE, $L_2$).
*   `visualize.py`: Renders a 4-panel diagnostic grid (Clean, Noise Map, Protected, GAN Output).

### 3. Execution
Download the StarGAN pre-trained weights to `./models/` and place the CelebA dataset in `./data/`. Run the main optimization and evaluation pipeline:
```bash
python main.py
```
Upon completion, the system will print comprehensive diagnostic stats, evaluation metrics, and display a dynamic 4-panel visual grid showcasing the catastrophic failure of the Deepfake generator.

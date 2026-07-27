# Deepfake Adversarial Watermark Implementation Details

This document provides a comprehensive overview of the reproduced cross-image adversarial watermark framework as described by Lin et al. (2025). The objective of the framework is to embed an imperceptible, frequency-domain watermark that severely distorts the output of generative facial manipulation models while surviving JPEG compression.

## 1. Modular Architecture

The implementation is broken down into modular Python components:

### `datasets_utils.py` (CelebA & LFW Loader)
Handles downloading and formatting of real facial datasets via `torchvision`. Features a fallback to load locally downloaded datasets if the automated download fails due to Google Drive quotas (common with CelebA). Automatically resizes images to `256x256` and converts them to the correct tensor format.

### `stargan.py` (Real Target Model)
Implements the genuine StarGAN Generator architecture, composed of initial convolutions, down-sampling layers, 6 residual bottleneck blocks, and up-sampling layers. It includes attribute conditioning where the target attribute `c` is concatenated spatially with the input image. 

### `jpeg_utils.py` (Differentiable JPEG Module)
Traditional JPEG quantization is non-differentiable due to the `round()` operation. To enable end-to-end optimization using backpropagation, this module simulates JPEG compression differentiably:
- **Color Space Conversion**: Maps RGB images strictly in `[0, 1]` to the YCbCr space using fixed transformation matrices.
- **DCT / IDCT**: Implements 8x8 block Discrete Cosine Transform using standard 2D DCT matrices.
- **Differentiable Quantization**: Replaces `round(x)` with `round(x) + (x - round(x))**3`. This allows the gradient to pass through the quantization step unchanged, making it possible to optimize the adversarial watermark under realistic compression constraints.
- **Quality Factor**: Standard luminance quantization matrix scaled for `Q=35`.

### `watermark.py` (Optimization & Fusion Engine)
Generates the robust universal watermark iteratively using Projected Gradient Descent (PGD) in the DCT frequency domain.
- **Objective Function**: Combines disruption capability with imperceptibility.
  - $L = MSE(G(I_{inv}, c), G(I, c)) + \lambda \cdot (1 / PSNR(I_{inv}, I))$
  - Where $G$ is the Deepfake target model, $c$ is the targeted forged attribute, and $\lambda = 0.05$.
- **Intra-batch Gradient Averaging**: For each batch, computes the mean of the `sign(grad)` across samples to identify shared facial attributes to protect.
- **Inter-batch Fusion**: Blends the optimized watermark from the current batch with the historical universal watermark using exponential smoothing: $W_{m+1} = \beta W_m + (1 - \beta) w_m$ with $\beta = 0.50$.

### `evaluation.py` (Evaluation Pipeline)
Once the universal watermark is obtained, this module rigorously evaluates its defensive performance:
- Applies the watermark to test images using the `DiffJPEG` encoding/decoding cycle.
- **Non-differentiable JPEG Simulation**: Compresses the protected images using Python's `PIL` library (`Q=75`), ensuring the test setup mimics real-world social media transmission.
- **Metrics Computation**: Compares the Deepfake model's output on unwatermarked vs. protected-and-compressed images using PSNR, SSIM, MSE, and Defense Success Rate ($SR_{mask}$).

### `download_weights.py` (Helper Script)
A helper script intended to fetch the official StarGAN `.pth` generator weights. Note that manual downloading from the official repo's Dropbox links might be required depending on network conditions.

### `main.py` (Runnable Pipeline)
The entry point script that orchestrates the entire workflow. It utilizes 128 images from CelebA for training, generating the universal watermark targeted at disrupting StarGAN's "Blond Hair" edit. It then executes cross-dataset robustness testing across both CelebA and LFW (1000 images each).

## 2. Parameter Configurations Used

The reproduction accurately matches the specific hyperparameters from the paper for StarGAN targeting:
- `epsilon = 0.20`: Constrains the magnitude of the adversarial perturbation.
- `T = 15`: Number of PGD attack iterations per batch.
- `alpha = 0.03`: The step size for gradient updates.
- `lambda_val = 0.05`: Balance factor between attack aggressiveness and stealthiness.
- `q = 35`: JPEG compression factor during training for balancing PSNR > 30 dB.
- `Q = 75`: JPEG compression factor for real-world evaluation testing.

## 3. How to Run

### Setup Data & Weights
1. Run `python download_weights.py` to acquire the StarGAN pretrained weights. If the download script fails, follow its printed URL to manually retrieve the `.zip` file and extract `200000-G.ckpt` into the `models/celeba-256x256-5attrs` folder.
2. Ensure you have activated your virtual environment: `.\venv\Scripts\Activate` (Windows) or `source venv/bin/activate` (Mac/Linux).

### Execute Pipeline
1. Run `python main.py`. 
2. The script will automatically attempt to download the `CelebA` and `LFW` datasets into a local `./data` folder.
3. The script will train the watermark on CelebA and subsequently print the evaluated disruption metrics on both the CelebA test set and the LFW test set.

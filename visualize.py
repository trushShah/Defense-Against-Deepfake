import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_watermark_results(I_clean, I_watermarked, G_clean, G_watermarked, psnr, ssim, watermark_grid=None):
    """
    Pops up a visual comparison window showing:
    1. Original Input Image
    2. Universal Watermark Pattern
    3. Watermarked/Protected Image
    4. GAN Output on Protected Image (Disrupted)
    """
    # Helper to convert PyTorch Tensor (C, H, W) to NumPy (H, W, C)
    def to_np(tensor):
        img = tensor.detach().cpu().squeeze(0)
        img = torch.clamp(img, 0.0, 1.0)
        return img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Proactive Deepfake Defense Evaluation\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}", fontsize=14, fontweight='bold')

    # 1. Clean Original Image
    axes[0].imshow(to_np(I_clean))
    axes[0].set_title("1. Original Image ($I_{\\text{clean}}$)")
    axes[0].axis("off")

    # 2. Watermark Noise Map
    diff = torch.abs(I_watermarked - I_clean)
    axes[1].imshow(to_np(diff * 10.0))
    axes[1].set_title("2. Watermark Noise Map\n($|I_{\\text{protected}} - I_{\\text{clean}}| \\times 10$)")
    axes[1].axis("off")

    # 3. Watermarked / Protected Image
    axes[2].imshow(to_np(I_watermarked))
    axes[2].set_title("3. Protected Image ($I_{\\text{protected}}$)")
    axes[2].axis("off")

    # 4. GAN Output on Protected Image
    axes[3].imshow(to_np(G_watermarked))
    axes[3].set_title("4. StarGAN Output\n($G(I_{\\text{protected}}, c)$)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
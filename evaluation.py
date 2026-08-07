import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np

def apply_jpeg_compression(image_tensor, quality=75):
    """
    Applies standard non-differentiable JPEG compression using PIL.
    Simulates real-world lossy compression.
    """
    B, C, H, W = image_tensor.shape
    compressed_tensors = []
    
    for i in range(B):
        img_np = image_tensor[i].detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        img_pil = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        img_compressed = Image.open(buffer)
        img_compressed_np = np.array(img_compressed).astype(np.float32) / 255.0
        compressed_tensor = torch.from_numpy(img_compressed_np).permute(2, 0, 1)
        compressed_tensors.append(compressed_tensor)
        
    return torch.stack(compressed_tensors).to(image_tensor.device)

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    
    img1_np = img1.detach().cpu().permute(0, 2, 3, 1).numpy()
    img2_np = img2.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    ssim_vals = []
    for i in range(img1_np.shape[0]):
        val = ssim(img1_np[i], img2_np[i], data_range=1.0, channel_axis=-1)
        ssim_vals.append(val)
    return np.mean(ssim_vals)

def evaluate_watermark(target_model, diff_jpeg, I_test, c_target, universal_watermark, Q=75, mask_threshold=0.05):
    """
    Evaluates the performance of the adversarial watermark on test images.
    """
    target_model.eval()
    # 1. Apply watermark in frequency domain
    
    I_protected = diff_jpeg(I_test, universal_watermark)
    
    # 2. Apply non-differentiable JPEG
    I_compressed = apply_jpeg_compression(I_protected, quality=Q)
    
    # 3. Pass through target model (StarGAN expects [-1, 1] input)
    with torch.no_grad():
        G_I_clean = target_model((I_test - 0.5) * 2.0, c_target)
        G_I_compressed = target_model((I_compressed - 0.5) * 2.0, c_target)
        
        # Keep G_I_clean and G_I_compressed in their native [-1, 1] scale
        # for authentic L2 distance evaluation against the 0.05 threshold.

    
    # 4. Metrics
    # Imperceptibility (Fidelity) Metrics
    psnr = calculate_psnr(I_protected, I_test).item()
    ssim = calculate_ssim(I_protected, I_test)
    
    # Global Disruption Metric (Full Image)
    mse = F.mse_loss(G_I_compressed, G_I_clean).item() # No change
    
    # Disruption Metrics over facial region (Center Crop)
    B, C, H, W = G_I_compressed.shape
    crop_h, crop_w = int(H * 0.15), int(W * 0.15)
    
    G_compressed_face = G_I_compressed[:, :, crop_h:-crop_h, crop_w:-crop_w]
    G_clean_face = G_I_clean[:, :, crop_h:-crop_h, crop_w:-crop_w]
    
    # Compute L2 disruption distance strictly on the facial region
    l2_diff = (G_compressed_face - G_clean_face) ** 2
    
    # Calculate the exact number of pixels in the bounding box
    face_pixels = C * (H - 2 * crop_h) * (W - 2 * crop_w)
    l2_mask = torch.sum(l2_diff, dim=[1, 2, 3]) / face_pixels

    sr_mask = (l2_mask > mask_threshold).float().mean().item() * 100.0
    
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MSE": mse,
        "L2_mask_avg": l2_mask.mean().item(),
        "SR_mask(%)": sr_mask,
        "Images": (I_test, I_compressed, G_I_clean, G_I_compressed)
    }

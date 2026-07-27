import torch
import torch.nn as nn
import numpy as np

def rgb_to_ycbcr(image):
    """
    Convert RGB image to YCbCr color space.
    Input image shape: (B, 3, H, W). Pixel values expected in [0, 1].
    Output shape: (B, 3, H, W).
    """
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=image.dtype, device=image.device)
    shift = torch.tensor([0.0, 0.5, 0.5], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    
    # image is (B, 3, H, W)
    # Permute to (B, H, W, 3) for matmul
    img_permuted = image.permute(0, 2, 3, 1)
    ycbcr = torch.matmul(img_permuted, matrix.t())
    ycbcr = ycbcr.permute(0, 3, 1, 2) + shift
    return ycbcr

def ycbcr_to_rgb(image):
    """
    Convert YCbCr image to RGB color space.
    Input image shape: (B, 3, H, W).
    Output shape: (B, 3, H, W).
    """
    matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=image.dtype, device=image.device)
    shift = torch.tensor([0.0, 0.5, 0.5], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    
    img_shifted = image - shift
    img_permuted = img_shifted.permute(0, 2, 3, 1)
    rgb = torch.matmul(img_permuted, matrix.t())
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb

class DiffJPEG(nn.Module):
    def __init__(self, quality_factor=35):
        super(DiffJPEG, self).__init__()
        self.q = quality_factor
        self.dct_scale = 1000.0  # Normalization factor for DCT coefficients
        self.build_dct_matrix()
        self.build_quantization_matrix()

    def build_dct_matrix(self):
        # 8x8 DCT matrix
        dct_m = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    dct_m[i, j] = 1 / np.sqrt(8)
                else:
                    dct_m[i, j] = np.sqrt(2 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
        self.register_buffer('dct_matrix', torch.from_numpy(dct_m))

    def build_quantization_matrix(self):
        # Standard JPEG luminance quantization matrix
        q_luminance = np.array([
            [16,  11,  10,  16,  24,  40,  51,  61],
            [12,  12,  14,  19,  26,  58,  60,  55],
            [14,  13,  16,  24,  40,  57,  69,  56],
            [14,  17,  22,  29,  51,  87,  80,  62],
            [18,  22,  37,  56,  68, 109, 103,  77],
            [24,  35,  55,  64,  81, 104, 113,  92],
            [49,  64,  78,  87, 103, 121, 120, 101],
            [72,  92,  95,  98, 112, 100, 103,  99]
        ], dtype=np.float32)

        # Scale by quality factor
        if self.q < 50:
            scale = 5000 / self.q
        else:
            scale = 200 - self.q * 2
            
        q_luminance = np.floor((q_luminance * scale + 50) / 100)
        q_luminance[q_luminance == 0] = 1
        q_luminance[q_luminance > 255] = 255
        self.register_buffer('q_matrix', torch.from_numpy(q_luminance))

    def dct_8x8(self, image):
        """
        Apply 8x8 DCT to the input image.
        Input: (B, C, H, W) where H and W are multiples of 8.
        """
        B, C, H, W = image.shape
        # Block split
        blocks = image.view(B, C, H // 8, 8, W // 8, 8)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5) # (B, C, H//8, W//8, 8, 8)
        
        # D * X * D^T
        dct_m = self.dct_matrix
        # X: (B, C, H//8, W//8, 8, 8)
        # dct_m: (8, 8)
        out = torch.matmul(dct_m, blocks) # D * X
        out = torch.matmul(out, dct_m.t()) # (D * X) * D^T
        return out

    def idct_8x8(self, blocks):
        """
        Apply 8x8 Inverse DCT to the input blocks.
        Input: (B, C, H//8, W//8, 8, 8)
        Output: (B, C, H, W)
        """
        B, C, H8, W8, _, _ = blocks.shape
        dct_m = self.dct_matrix
        # D^T * Y * D
        out = torch.matmul(dct_m.t(), blocks)
        out = torch.matmul(out, dct_m)
        
        # Reconstruct image
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous()
        out = out.view(B, C, H8 * 8, W8 * 8)
        return out

    def quantize(self, dct_blocks):
        """
        Differentiable quantization using Straight-Through Estimator (STE).
        Provides much stronger gradients than cubic approximations.
        """
        x = dct_blocks / self.q_matrix
        x_approx = x + (torch.round(x) - x).detach()
        return x_approx * self.q_matrix

    def encode(self, x):
        """
        x: RGB image [0, 1], shape (B, 3, H, W)
        Returns: YCbCr DCT coefficients. shape (B, 3, H//8, W//8, 8, 8)
        """
        ycbcr = rgb_to_ycbcr(x)
        ycbcr = (ycbcr - 0.5) * 255 # Scale to [-128, 127] roughly for DCT
        dct_blocks = self.dct_8x8(ycbcr)
        return dct_blocks

    def decode(self, dct_blocks):
        """
        dct_blocks: DCT coefficients
        Returns: RGB image [0, 1]
        """
        ycbcr = self.idct_8x8(dct_blocks)
        ycbcr = ycbcr / 255 + 0.5
        rgb = ycbcr_to_rgb(ycbcr)
        return torch.clamp(rgb, 0, 1)

    def forward(self, x, watermark_dct_y=None):
        """
        Applies differentiable JPEG compression.
        Optionally adds a watermark to the Y channel DCT coefficients.
        """
        dct_blocks = self.encode(x)
        
        if watermark_dct_y is not None:
            # watermark_dct_y should have shape (1, 1, H//8, W//8, 8, 8) or matching (B, ...)
            # Add to the Y channel (index 0)
            dct_y = dct_blocks[:, 0:1, :, :, :, :]
            
            # Normalize DCT coefficients to approx [-1, 1] range
            dct_y_norm = dct_y / self.dct_scale
            
            # Add watermark w in the normalized space
            dct_y_norm = dct_y_norm + watermark_dct_y
            
            # Denormalize back to original DCT scale
            dct_y_watermarked = dct_y_norm * self.dct_scale
            
            dct_blocks = torch.cat([dct_y_watermarked, dct_blocks[:, 1:, :, :, :, :]], dim=1)
            
        quantized = self.quantize(dct_blocks)
        decoded = self.decode(quantized)
        return decoded

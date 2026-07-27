import torch
import torch.nn as nn

class SyntheticDeepfakeModel(nn.Module):
    """
    A synthetic/dummy Deepfake model simulating a generator G(x, c).
    In a real scenario, this would be StarGAN, AttGAN, or HiSD.
    """
    def __init__(self):
        super(SyntheticDeepfakeModel, self).__init__()
        # A simple convolutional block to mimic feature transformations
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        """
        Forward pass mimicking the forgery process.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            c (torch.Tensor, optional): Target attribute label (not used in this dummy model).
            
        Returns:
            torch.Tensor: Forged image tensor of shape (B, 3, H, W).
        """
        # Apply the synthetic transformation
        return self.net(x)

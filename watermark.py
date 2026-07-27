import torch
import torch.nn.functional as F

class WatermarkOptimizer:
    def __init__(self, target_model, diff_jpeg, epsilon=0.018, alpha=0.03, T=20, lambda_val=0.10):
        self.target_model = target_model
        self.diff_jpeg = diff_jpeg
        self.epsilon = epsilon
        self.alpha = alpha
        self.T = T
        self.lambda_val = lambda_val
        
        # Mid-frequency mask
        self.mask = torch.zeros((1, 1, 1, 1, 8, 8))
        for u in range(8):
            for v in range(8):
                if 2 <= u + v <= 6:
                    self.mask[0, 0, 0, 0, u, v] = 1.0

    def compute_loss(self, I_clean, I_inv, c_target):
            # Original forgery G(I) (StarGAN expects [-1, 1])
            with torch.no_grad():
                G_I = self.target_model((I_clean - 0.5) * 2.0, c_target)
                
            # Distorted forgery G(I_inv)
            G_I_inv = self.target_model((I_inv - 0.5) * 2.0, c_target)
            
            # 1. Disruption loss (MSE between generator outputs)
            mse_disrupt = F.mse_loss(G_I_inv, G_I)
            
            # 2. Imperceptibility loss (MSE between watermarked and original)
            mse_distortion = F.mse_loss(I_inv, I_clean)
            
            # Total Loss: Maximize disruption while penalizing image distortion
            loss = mse_disrupt - self.lambda_val * mse_distortion
            return loss

    def optimize_batch(self, I_clean, c_target, w_init=None):
        """
        Runs PGD optimization on a single batch.
        Returns the optimized batch-specific watermark.
        """
        B, C, H, W = I_clean.shape
        
        if w_init is None:
            # Random initialization in DCT space
            w = torch.empty((B, 1, H // 8, W // 8, 8, 8), device=I_clean.device).uniform_(-self.epsilon, self.epsilon)
            mask = self.mask.to(I_clean.device)
            w = w * mask
        else:
            w = w_init.clone().detach().to(I_clean.device)
            if w.shape[0] == 1:
                w = w.repeat(B, 1, 1, 1, 1, 1)

        w.requires_grad = True

        for t in range(self.T):
            I_inv = self.diff_jpeg(I_clean, w)
            loss = self.compute_loss(I_clean, I_inv, c_target)
            
            self.target_model.zero_grad()
            self.diff_jpeg.zero_grad()
            
            grad = torch.autograd.grad(loss, w)[0]
            
            # Intra-batch Gradient Averaging: grad_avg = torch.mean(torch.sign(grad), dim=0)
            grad_avg = torch.mean(torch.sign(grad), dim=0, keepdim=True)
            mask = self.mask.to(I_clean.device)
            w = w + self.alpha * grad_avg * mask
            w = torch.clamp(w, -self.epsilon, self.epsilon) * mask
            
            w = w.detach()
            w.requires_grad = True

        # Return the batch-averaged watermark as the local batch watermark w^m
        return w.mean(dim=0, keepdim=True).detach()

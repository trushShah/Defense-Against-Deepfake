import torch
from stargan import StarGAN_Generator
from jpeg_utils import DiffJPEG
from watermark import WatermarkOptimizer
from evaluation import evaluate_watermark
from datasets_utils import get_datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from visualize import visualize_watermark_results
import os

def load_stargan_weights(model, path="models/celeba-256x256-5attrs/200000-G.ckpt"):
    if os.path.exists(path):
        print(f"Loading StarGAN weights from {path}...")
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    else:
        print(f"WARNING: StarGAN weights not found at {path}.")
        print("Please run `python download_weights.py` to download them.")
        print("Proceeding with randomly initialized weights (Defense metrics will not be meaningful!).")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Models
    # StarGAN: 5 target attributes for CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young)
    model = StarGAN_Generator(c_dim=5).to(device)
    load_stargan_weights(model)
    model.eval() # Generator should be in eval mode
    
    diff_jpeg = DiffJPEG(quality_factor=35).to(device)
    
    # Target: StarGAN specific settings from the paper
    epsilon = 0.021
    alpha = 0.004
    T = 25
    lambda_val = 0.08
    
    optimizer = WatermarkOptimizer(
        target_model=model, 
        diff_jpeg=diff_jpeg, 
        epsilon=epsilon, 
        alpha=alpha, 
        T=T, 
        lambda_val=lambda_val
    )
    
    # 2. Setup Real Datasets
    batch_size = 8
    print("Loading CelebA and LFW datasets...")
    celeba_train_set, celeba_test_set, lfw_test_set = get_datasets(root_dir="./data")
    
    # Use 128 training images (16 batches of size 8)
    train_size = 128
    if celeba_train_set is not None:
        train_subset = Subset(celeba_train_set, range(train_size))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    else:
        print(f"CelebA dataset unavailable. Generating {train_size} synthetic images for demonstration.")
        train_loader = [(torch.rand((batch_size, 3, 256, 256)), torch.zeros((batch_size, 5))) for _ in range(train_size // batch_size)]

    test_size = 1000 if device.type == 'cuda' else 64
    if celeba_test_set is not None:
        test_subset_celeba = Subset(celeba_test_set, range(test_size))
        test_loader_celeba = DataLoader(test_subset_celeba, batch_size=batch_size, shuffle=False)
    else:
        test_loader_celeba = [(torch.rand((10, 3, 256, 256)), torch.zeros((10, 5)))]
        
    if lfw_test_set is not None:
        test_subset_lfw = Subset(lfw_test_set, range(test_size))
        test_loader_lfw = DataLoader(test_subset_lfw, batch_size=batch_size, shuffle=False)
    else:
        test_loader_lfw = [(torch.rand((10, 3, 256, 256)), torch.zeros((10, 5)))]

    # 3. Training Loop (Adversarial Watermark Optimization)
    beta = 0.50
    universal_watermark = None
    
    print("Starting adversarial watermark optimization...")
    # Define active target attribute vector (e.g., Blond Hair + Male)
    c_target = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]).to(device)

    for i, data in enumerate(tqdm(train_loader, desc="Batches")):
        if isinstance(data, (list, tuple)):
            batch_img = data[0].to(device)
        else:
            batch_img = data.to(device)

        c_target_batch = c_target.repeat(batch_img.size(0), 1)

        w_init = universal_watermark
            
        w_batch = optimizer.optimize_batch(batch_img, c_target_batch, w_init=w_init)
        
        # Inter-batch Exponential Fusion
        if universal_watermark is None:
            universal_watermark = w_batch
        else:
            universal_watermark = beta * universal_watermark + (1 - beta) * w_batch
            
    print("Training complete. Universal watermark generated.")
    
    print("\n--- Diagnostic Stats ---")
    print("w_universal shape:", universal_watermark.shape)
    print("w_universal min/max:", universal_watermark.min().item(), universal_watermark.max().item())
    print("Non-zero frequency coefficients count:", (universal_watermark != 0).sum().item())
    print("------------------------\n")
    
    # 4. Evaluation Helper
    def run_evaluation(loader, dataset_name):
        Q = 75 # Standard social media compression
        print(f"\nEvaluating robust watermark on {dataset_name} (JPEG Q={Q})...")
        
        # Accumulate metrics
        total_metrics = {"PSNR": 0.0, "SSIM": 0.0, "MSE": 0.0, "L2_mask_avg": 0.0, "SR_mask(%)": 0.0}
        num_batches = 0
        
        for data in loader:
            if isinstance(data, (list, tuple)):
                batch_img = data[0].to(device)
            else:
                batch_img = data.to(device)
                
            c_target_batch = c_target.repeat(batch_img.size(0), 1)
            
            results = evaluate_watermark(
                target_model=model, 
                diff_jpeg=diff_jpeg, 
                I_test=batch_img, 
                c_target=c_target_batch,
                universal_watermark=universal_watermark, 
                Q=Q
            )
            
            for k in total_metrics.keys():
                total_metrics[k] += results[k]
                
            # Visualize the very first batch of CelebA
            if num_batches == 0 and dataset_name == "CelebA":
                I_clean, I_wm, G_clean, G_wm = results["Images"]
                
                print("I_test min/max:", I_clean.min().item(), I_clean.max().item())
                print("I_protected min/max:", I_wm.min().item(), I_wm.max().item())
                print("------------------------\n")
                
                # Pass just the first image in the batch for visualization
                visualize_watermark_results(
                    I_clean[0:1], 
                    I_wm[0:1], 
                    G_clean[0:1], 
                    G_wm[0:1], 
                    results["PSNR"], 
                    results["SSIM"], 
                    watermark_grid=universal_watermark
                )
                
            num_batches += 1
            
        print(f"\n--- {dataset_name} Evaluation Results ---")
        for k, v in total_metrics.items():
            print(f"{k}: {v / num_batches:.4f}")

    # 5. Run Evaluations
    run_evaluation(test_loader_celeba, "CelebA")
    run_evaluation(test_loader_lfw, "LFW (Cross-dataset)")


if __name__ == "__main__":
    main()

import os
import urllib.request
import zipfile

def download_stargan_weights(save_dir="./models"):
    """
    Downloads the official pretrained StarGAN weights for CelebA (256x256, 5 attributes).
    The weights are officially hosted on Dropbox by the authors.
    """
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, "celeba-256x256-5attrs.zip")
    extract_dir = os.path.join(save_dir, "celeba-256x256-5attrs")
    
    url = "https://www.dropbox.com/s/96ndvw2zhg2127g/celeba-256x256-5attrs.zip?dl=1"
    
    if os.path.exists(os.path.join(extract_dir, "200000-G.ckpt")):
        print(f"Weights already found at {extract_dir}/200000-G.ckpt")
        return

    print(f"Downloading StarGAN weights from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        print(f"Extracting to {save_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
            
        print(f"Extraction complete. Weights available in {extract_dir}")
        os.remove(zip_path) # cleanup
    except Exception as e:
        print(f"Failed to download or extract weights: {e}")
        print("Please download manually from https://github.com/yunjey/stargan")

if __name__ == "__main__":
    download_stargan_weights()

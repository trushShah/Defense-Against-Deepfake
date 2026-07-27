import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import csv

class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Check image directory
        self.image_dir = os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba')
        if not os.path.exists(self.image_dir):
             self.image_dir = os.path.join(root_dir, 'img_align_celeba')
        
        attr_path = os.path.join(root_dir, 'list_attr_celeba.csv')
        eval_path = os.path.join(root_dir, 'list_eval_partition.csv')
        
        if not os.path.exists(attr_path) or not os.path.exists(eval_path):
            raise FileNotFoundError("Missing CelebA CSV files. Ensure you have Kaggle's CelebA .csv files in data/celeba/")
            
        partition_val = '0' if split == 'train' else '2'
        
        # Read eval partition
        valid_images = set()
        with open(eval_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if row[1] == partition_val:
                    valid_images.add(row[0])
                    
        # Read attributes
        self.data = []
        target_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        attr_indices = []
        
        with open(attr_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for attr in target_attrs:
                if attr in header:
                    attr_indices.append(header.index(attr))
                else:
                    raise ValueError(f"Attribute {attr} not found in {attr_path}")
            
            for row in reader:
                if row[0] in valid_images:
                    # extract attributes and convert -1 to 0
                    labels = [1.0 if row[idx] == '1' else 0.0 for idx in attr_indices]
                    self.data.append((row[0], labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, labels = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32)

def get_datasets(root_dir="./data"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    celeba_train, celeba_test, lfw_test = None, None, None
    celeba_root = os.path.join(root_dir, 'celeba')
    lfw_root = os.path.join(root_dir, 'lfw-py', 'lfw-deepfunneled')
    
    if not os.path.exists(lfw_root):
        lfw_root = os.path.join(root_dir, 'lfw-py')

    try:
        celeba_train = CustomCelebADataset(root_dir=celeba_root, split='train', transform=transform)
        celeba_test = CustomCelebADataset(root_dir=celeba_root, split='test', transform=transform)
        print(f"Successfully loaded Custom CelebA dataset. Train: {len(celeba_train)}, Test: {len(celeba_test)}")
    except Exception as e:
        print(f"Failed to load CelebA: {e}")

    try:
        # LFW is just an ImageFolder for our testing purpose
        # root=lfw-py makes torchvision find classes inside (like lfw-deepfunneled or individual people)
        lfw_dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'lfw-py'), transform=transform)
        
        class LFWWrapper(Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                img, _ = self.ds[idx]
                return img, torch.zeros(5) # dummy 5 attributes
        
        lfw_test = LFWWrapper(lfw_dataset)
        print(f"Successfully loaded Custom LFW dataset. Test: {len(lfw_test)}")
    except Exception as e:
        print(f"Failed to load LFW: {e}")

    return celeba_train, celeba_test, lfw_test

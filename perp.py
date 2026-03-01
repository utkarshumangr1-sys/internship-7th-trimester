import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    TRAIN_IMAGE_PATH = r"C:\Users\pc\Desktop\3_b_bounadary_mask\Aug2Img"
    TRAIN_MASK_PATH  = r"C:\Users\pc\Desktop\3_b_bounadary_mask\Aug2masks"
    TEST_IMAGE_PATH  = r"C:\Users\pc\Desktop\bm_test\test_images"
    TEST_MASK_PATH   = r"C:\Users\pc\Desktop\bm_test\test_masks"
    OUTPUT_DIR       = r"C:\Users\pc\Desktop\correction"
    
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 2
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0

# ==================== BULLETPROOF COLLATE FUNCTION ====================
def collate_fn(batch):
    """🔥 FIXES ALL SIZE ISSUES - Normalizes dimensions + resizes"""
    images = []
    masks = []
    
    for img, mask in batch:
        # ✅ FORCE CORRECT DIMENSIONS
        # Images: Always (3, H, W)
        if img.dim() == 3 and img.size(0) not in [1, 3]:
            img = img.permute(2, 0, 1)  # HWC -> CHW
        elif img.dim() == 3 and img.size(0) == 1:
            img = img.repeat(3, 1, 1)   # Grayscale -> RGB
        
        # Masks: Always (1, H, W)
        if mask.dim() == 3 and mask.size(0) not in [1]:
            mask = mask.unsqueeze(0)    # HW -> 1HW
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)    # HW -> 1HW
        
        # ✅ RESIZE TO FIXED SIZE
        img_resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=Config.IMG_SIZE, mode='bilinear', align_corners=False
        ).squeeze(0)  # (3,512,512)
        
        mask_resized = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=Config.IMG_SIZE, mode='nearest'
        ).squeeze(0)  # (1,512,512)
        
        images.append(img_resized)
        masks.append(mask_resized)
    
    return torch.stack(images), torch.stack(masks)  # ✅ ALL SAME SIZE NOW

# ==================== GEOTIFF DATASET ====================
class BoundaryGeoTIFFDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_files = sorted(self.image_dir.glob('*.tif*'))
        print(f"✅ Found {len(self.image_files)} image-mask pairs")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .tif files in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        # Load raw data
        image = tifffile.imread(str(img_path))
        mask = tifffile.imread(str(mask_path))
        
        # Handle mask dimensions
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to tensors (let collate_fn handle dimensions)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return image, mask

# ==================== MODEL ====================
class BinaryDeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model = model
    
    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])

# ==================== METRICS ====================
class Metrics:
    @staticmethod
    def dice(pred, target, smooth=1e-6):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    @staticmethod
    def iou(pred, target, smooth=1e-6):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

# ==================== TRAINER ====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        
        self.model = BinaryDeepLabV3().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        print(f"🚀 DeepLabV3-ResNet50 on {self.device}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        
        pbar = tqdm(dataloader, desc='Train')
        for images, masks in pbar:
            images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, masks)
            loss.backward()
            self.optimizer.step()
            
            dice = Metrics.dice(preds, masks)
            iou = Metrics.iou(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'IoU': f'{iou.item():.4f}'
            })
        
        return total_loss/len(dataloader), total_dice/len(dataloader), total_iou/len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss, total_dice, total_iou = 0, 0, 0
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc='Val'):
                images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
                preds = self.model(images)
                loss = self.criterion(preds, masks)
                
                dice = Metrics.dice(preds, masks)
                iou = Metrics.iou(preds, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
        
        return total_loss/len(dataloader), total_dice/len(dataloader), total_iou/len(dataloader)
    
    def train(self, train_loader, val_loader):
        best_dice = 0
        history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            print("="*50)
            
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            val_loss, val_dice, val_iou = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            
            print(f"Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}, IoU={train_iou:.4f}")
            print(f"Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), f'{self.config.OUTPUT_DIR}/best_model.pth')
                print(f"💾 Saved best model (Dice: {best_dice:.4f})")
        
        torch.save(self.model.state_dict(), f'{self.config.OUTPUT_DIR}/final_model.pth')
        pd.DataFrame(history).to_csv(f'{self.config.OUTPUT_DIR}/training_history.csv')
        self.plot_history(history)
    
    def plot_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[1].plot(history['train_dice'], label='Train Dice')
        axes[1].plot(history['val_dice'], label='Val Dice')
        axes[1].set_title('Dice Score')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_DIR}/training_curves.png', dpi=300)
        plt.close()

# ==================== MAIN ====================
def main():
    config = Config()
    
    print("📁 Loading datasets...")
    train_dataset = BoundaryGeoTIFFDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_MASK_PATH)
    test_dataset = BoundaryGeoTIFFDataset(config.TEST_IMAGE_PATH, config.TEST_MASK_PATH)
    
    train_size = int((1 - config.VAL_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, 
                            shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"✅ Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")
    
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    
    print(f"\n🎉 Done! Results: {config.OUTPUT_DIR}")

if __name__ == '__main__':
    main()

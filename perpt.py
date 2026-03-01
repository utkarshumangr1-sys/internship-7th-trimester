import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ==================== MODEL CLASS (Same as Training) ====================
class BinaryDeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model = model
    
    def forward(self, x):
        output = self.model(x)['out']
        return torch.sigmoid(output)  # Binary probabilities [0,1]

# ==================== PREDICTION CLASS ====================
class BoundaryPredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        print(f"🚀 Using device: {self.device}")
        
        # Load trained model
        self.model = BinaryDeepLabV3().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded from: {model_path}")
    
    def predict_single(self, image_path, output_dir="predictions", threshold=0.9):
        """Predict boundary mask for single RGB GeoTIFF"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load image
        image = tifffile.imread(image_path)
        orig_shape = image.shape
        
        print(f"📸 Input: {image_path}")
        print(f"   Shape: {image.shape}, Range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Ensure CHW format (3,H,W)
        if image.shape[0] not in [1, 3]:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)  # [1,3,H,W]
        
        # Predict
        with torch.no_grad():
            pred = self.model(image_tensor)  # [1,1,H,W]
            pred_np = pred.squeeze().cpu().numpy()  # [H,W]
        
        # Apply threshold
        boundary_mask = (pred_np > threshold).astype(np.float32)
        
        # Save results
        base_name = Path(image_path).stem
        tifffile.imwrite(f"{output_dir}/{base_name}_boundary.tif", boundary_mask)
        tifffile.imwrite(f"{output_dir}/{base_name}_probability.tif", pred_np)
        
        # Visualize
        self._visualize(image, pred_np, boundary_mask, orig_shape, output_dir, base_name)
        
        print(f"✅ Saved: {output_dir}/{base_name}_boundary.tif")
        print(f"   Boundary pixels: {np.sum(boundary_mask):,}/{boundary_mask.size:,} ({100*np.mean(boundary_mask):.1f}%)")
        
        return boundary_mask, pred_np
    
    def _visualize(self, image, pred_prob, boundary_mask, orig_shape, output_dir, base_name):
        """Create nice visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        if image.shape[0] == 3:  # CHW
            img_vis = np.transpose(image, (1, 2, 0))
        else:
            img_vis = image
        
        axes[0,0].imshow(img_vis)
        axes[0,0].set_title('Original RGB')
        axes[0,0].axis('off')
        
        # Probability heatmap
        im1 = axes[0,1].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[0,1].set_title('DeepLabV3_ResNet50 Boundary Probability')
        plt.colorbar(im1, ax=axes[0,1])
        
        # Binary mask
        axes[1,0].imshow(boundary_mask, cmap='gray')
        axes[1,0].set_title('Binary Boundary (Threshold=0.9)')
        axes[1,0].axis('off')
        
        # Overlay
        overlay = img_vis.copy()
        overlay[boundary_mask == 1] = [1, 0, 0]  # Red boundaries
        axes[1,1].imshow(overlay)
        axes[1,1].set_title('Boundaries Overlay (RED)')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization: {output_dir}/{base_name}_results.png")

# ==================== BATCH PREDICTION ====================
def predict_folder(predictor, image_folder, output_dir="batch_predictions"):
    """Predict on entire folder"""
    Path(output_dir).mkdir(exist_ok=True)
    
    image_paths = sorted(Path(image_folder).glob('*.tif*'))
    print(f"\n🔥 Batch predicting {len(image_paths)} images...")
    
    all_boundaries = []
    for img_path in tqdm(image_paths, desc="Predicting"):
        boundary, prob = predictor.predict_single(str(img_path), output_dir)
        all_boundaries.append(boundary)
    
    print(f"✅ Batch complete! All results in: {output_dir}")

# ==================== MAIN USAGE ====================
if __name__ == '__main__':
    # === SINGLE IMAGE PREDICTION ===
    MODEL_PATH = r"C:\Users\pc\Desktop\correction\DeeplabV3-Resnet50-predictions\final_model.pth"
    
    predictor = BoundaryPredictor(MODEL_PATH)
    
    # Predict single image
    IMAGE_PATH = r"C:\Users\pc\Desktop\bm_test\test_images\Img_Site12.tif"  # ← UPDATE THIS
    boundary_mask, probability_map = predictor.predict_single(IMAGE_PATH)
    
    # === BATCH PREDICTION (uncomment) ===
    # predict_folder(predictor, r"C:\path\to\image_folder")
    
    print("\n🎉 Prediction complete!")
    print("Files generated:")
    print("  - *_boundary.tif     → Binary mask (0/1)")
    print("  - *_probability.tif  → Confidence map [0,1]") 
    print("  - *_results.png      → Visualization")

# Land Boundary Detection using DeepLabV3-ResNet50

Farmland boundary detection from satellite imagery using DeepLabV3-ResNet50. Dice score: 0.9841.

---

## Files

| File | Description |
|------|-------------|
| `perp.py` | Training script — loads GeoTIFF images and masks, trains DeepLabV3-ResNet50 model, saves weights and training curves |
| `perpt.py` | Prediction script — loads the trained model and runs inference on a satellite image, outputs probability map, binary mask, and visualization |
| `training_curves.png` | Loss and Dice score curves over 50 training epochs |
| `Img_Site12_results.png` | Inference result on test site — RGB input, probability heatmap, binary boundary mask, and overlay |

---

## Model

- **Architecture:** DeepLabV3 with ResNet-50 backbone
- **Pretrained:** ImageNet weights
- **Task:** Binary semantic segmentation — boundary vs. non-boundary
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam, lr = 1e-4
- **Epochs:** 50
- **Inference threshold:** 0.9
- **Validation Dice:** 0.9841

---

## Internship Project
**IIT Guwahati** — School of Agro and Rural Technology
Under the supervision of **Dr. Dipankar Mandal**

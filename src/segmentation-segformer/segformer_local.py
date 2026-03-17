import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 2. Load Model
# We use the official Cityscapes fine-tuned checkpoint
repo_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(repo_name)
model = SegformerForSemanticSegmentation.from_pretrained(repo_name).to(device)

# 3. Load LOCAL Image
# Make sure testimage.png is in the same folder as this script
image_path = r"testimage4.png" 
try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded {image_path}")
except FileNotFoundError:
    print(f"Error: Could not find {image_path}. Please check the filename.")
    exit()

# 4. Inference
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. Upsample logits to original image size
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (H, W)
    mode='bilinear',
    align_corners=False
)

# 6. Get the class labels
pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

# 7. Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original Image
axs[0].imshow(image)
axs[0].set_title("Input Image")
axs[0].axis('off')

# Segmentation Mask
# 'tab20' is a good colormap for categorical data (distinct colors for classes)
axs[1].imshow(pred_seg, cmap='tab20')
axs[1].set_title("SegFormer Prediction")
axs[1].axis('off')

plt.tight_layout()
plt.show()

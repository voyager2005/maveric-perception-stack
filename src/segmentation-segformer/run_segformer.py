import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 1. Load Model (SegFormer B1 is a good balance for experiments)
# We use the official Cityscapes fine-tuned checkpoint
repo_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(repo_name)
model = SegformerForSemanticSegmentation.from_pretrained(repo_name).to(device)

# 2. Get Test Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Placeholder; replace with Cityscapes local file if available
image = Image.open(requests.get(url, stream=True).raw)

# 3. Inference
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# 4. Upsample to original size
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (H, W)
    mode='bilinear',
    align_corners=False
)
pred_seg = upsampled_logits.argmax(dim=1)[0]

# 5. Show
plt.imshow(pred_seg.cpu().numpy())
plt.show()

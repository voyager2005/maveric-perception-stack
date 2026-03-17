import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm # Progress bar

# --- Configuration ---
INPUT_FOLDER = r"/home/cv/Documents/SegTransformer/Data/Images/"
OUTPUT_VIDEO_PATH = r"/home/cv/Documents/SegTransformer/Data/segmentation_output.mp4"
MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024" # The "Big" Model
FPS = 24 # Must match your extraction FPS

# Official Cityscapes Color Palette (r, g, b)
# This maps class indices (0-18) to colors
PALETTE = np.array([
    [128, 64, 128],  # 0: road (purple)
    [244, 35, 232],  # 1: sidewalk (pink)
    [70, 70, 70],    # 2: building (grey)
    [102, 102, 156], # 3: wall
    [190, 153, 153], # 4: fence
    [153, 153, 153], # 5: pole
    [250, 170, 30],  # 6: traffic light (orange)
    [220, 220, 0],   # 7: traffic sign (yellow)
    [107, 142, 35],  # 8: vegetation (green)
    [152, 251, 152], # 9: terrain
    [70, 130, 180],  # 10: sky (blue)
    [220, 20, 60],   # 11: person (red)
    [255, 0, 0],     # 12: rider
    [0, 0, 142],     # 13: car (dark blue)
    [0, 0, 70],      # 14: truck
    [0, 60, 100],    # 15: bus
    [0, 80, 100],    # 16: train
    [0, 0, 230],     # 17: motorcycle
    [119, 11, 32],   # 18: bicycle
    [0, 0, 0]        # 19: void/background
], dtype=np.uint8)

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device} (Model: {MODEL_ID})")

    # 2. Load Model & Processor
    print("Loading SegFormer B5...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # 3. Get List of Images
    # We sort them to ensure video frames are in order
    image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    # 4. Setup Video Writer
    # Read first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape
    
    # 'mp4v' is the most compatible codec for .mp4 on Linux OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))

    print(f"Processing {len(image_files)} frames...")

    # 5. Processing Loop
    for img_path in tqdm(image_files):
        # A. Load Image (PIL for Model)
        image_pil = Image.open(img_path).convert("RGB")
        
        # B. Inference
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # C. Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # D. Get Prediction Mask
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # E. Colorize Mask
        # Map class indices to RGB colors
        color_mask = PALETTE[pred_seg]
        
        # F. Overlay (Alpha Blending)
        # Convert PIL image to OpenCV format (RGB -> BGR)
        original_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        color_mask_cv = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        
        # Blend: 50% original, 50% mask
        blended_frame = cv2.addWeighted(original_cv, 0.5, color_mask_cv, 0.5, 0)
        
        # G. Write to Video
        video_writer.write(blended_frame)

    # 6. Cleanup
    video_writer.release()
    print(f"\nSuccess! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()

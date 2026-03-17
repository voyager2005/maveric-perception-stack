import os
import cv2
import numpy as np
import glob
import time
from trt_segformer import SegFormerTRT # Ensure this matches your filename

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
ENGINE_PATH = "segformer_b1_python_build.engine"
BASE_DIR = "/home/cv/Documents/points/Data/v1.0-mini/sweeps"

# Your specific camera sweeps
SWEEP_FOLDERS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT"
]

# Cityscapes 19-class color palette (Converted to BGR for OpenCV)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # 0: road
    [232, 35, 244],   # 1: sidewalk 
    [70, 70, 70],     # 2: building
    [156, 102, 102],  # 3: wall
    [153, 153, 190],  # 4: fence
    [153, 153, 153],  # 5: pole
    [30, 170, 250],   # 6: traffic light
    [0, 220, 220],    # 7: traffic sign
    [35, 142, 107],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [180, 130, 70],   # 10: sky
    [60, 20, 220],    # 11: person
    [0, 0, 255],      # 12: rider
    [142, 0, 0],      # 13: car
    [70, 0, 0],       # 14: truck
    [100, 60, 0],     # 15: bus
    [100, 80, 0],     # 16: train
    [230, 0, 0],      # 17: motorcycle
    [32, 11, 119]     # 18: bicycle
], dtype=np.uint8)
# ==========================================

def process_sweeps():
    print(f"Loading TensorRT Model from: {ENGINE_PATH}...")
    trt_model = SegFormerTRT(ENGINE_PATH)

    for folder in SWEEP_FOLDERS:
        input_dir = os.path.join(BASE_DIR, folder)
        output_dir = os.path.join(BASE_DIR, f"{folder}_MASK")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Grab all .jpg or .png images in the input directory
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        
        if not image_paths:
            print(f"⚠️ Warning: No images found in {input_dir}")
            continue

        print(f"\nProcessing {len(image_paths)} images from {folder}...")
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"❌ Failed to read {filename}")
                continue

            h, w = img.shape[:2]

            # 1. Run Inference 
            logits, _ = trt_model.infer(img)
            
            # 2. Extract Mask (Matches the logic from your calibration script)
            seg_mask_ids = np.argmax(logits, axis=1)[0]
            
            # 3. Resize Mask back to original image dimensions
            mask_resized = cv2.resize(seg_mask_ids.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 4. Convert Class IDs to Colors via NumPy indexing (Super fast)
            color_mask = CITYSCAPES_COLORS[mask_resized]
            
            # 5. Save the colored mask
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, color_mask)
            
            # Simple progress tracker
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images...")

        elapsed = time.time() - start_time
        print(f"✅ Finished {folder} in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    process_sweeps()

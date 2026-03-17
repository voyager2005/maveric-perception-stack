import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
ENGINE_PATH = "yolov8n-obb.engine"
IMAGE_PATH = "bev_semantic_sanity_check.png" # The BEV map we generated earlier
OUTPUT_PATH = "bev_obb_result.jpg"
# ==========================================

def run_inference():
    print(f"Loading TensorRT Engine: {ENGINE_PATH}...")
    # Ultralytics natively handles the TensorRT runtime backend
    model = YOLO(ENGINE_PATH, task='obb')
    
    print(f"Running inference on {IMAGE_PATH}...")
    # Run prediction. We set a low confidence threshold just to see what the zero-shot model picks up.
    results = model.predict(source=IMAGE_PATH, conf=0.1, save=False)
    
    # Process the results (we only passed one image, so we grab the first result)
    result = results[0]
    
    # 1. Automatic Visualization
    # The plot() method draws the rotated boxes, labels, and confidences natively
    annotated_img = result.plot()
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    print(f"\n✅ Saved annotated visualization to {OUTPUT_PATH}")
    
    # 2. Extracting the Raw Math for our 3D Pipeline
    if result.obb is not None:
        boxes = result.obb.xywhr.cpu().numpy() # [x_center, y_center, width, height, rotation(rad)]
        classes = result.obb.cls.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        
        print(f"\nDetected {len(boxes)} Oriented Bounding Boxes:")
        for i in range(len(boxes)):
            x, y, w, h, r = boxes[i]
            cls_id = int(classes[i])
            conf = confidences[i]
            print(f"  [Box {i}] Class: {cls_id} | Conf: {conf:.2f} | X: {x:.1f}, Y: {y:.1f}, W: {w:.1f}, H: {h:.1f}, Angle: {r:.2f} rad")
    else:
        print("\n⚠️ No bounding boxes detected above the confidence threshold.")

if __name__ == "__main__":
    run_inference()

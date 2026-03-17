from ultralytics import YOLO
import numpy as np

class YoloTRT:
    def __init__(self, engine_path="yolov8n.engine", conf_thresh=0.3):
        """
        Wraps the YOLO TensorRT engine for standardized inference.
        """
        self.conf_thresh = conf_thresh
        print(f"Loading YOLO TRT Engine: {engine_path}")
        
        # YOLO natively routes .engine files to its TensorRT execution backend
        self.model = YOLO(engine_path, task='detect')
        
        # Warm up the TensorRT engine (first inference is always slow)
        print("Warming up YOLO TensorRT engine...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)
        print("✅ YOLO Engine Ready.")

    def infer(self, img):
        """
        Runs inference and returns bounding boxes and class IDs.
        Returns:
            boxes: [N, 4] array of [x1, y1, x2, y2]
            classes: [N] array of class IDs
        """
        # verbose=False prevents the console from being spammed every frame
        results = self.model(img, conf=self.conf_thresh, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        return boxes, classes

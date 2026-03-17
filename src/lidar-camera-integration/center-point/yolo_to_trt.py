from ultralytics import YOLO

def export_to_tensorrt():
    print("Loading YOLOv8 Nano PyTorch Model...")
    # This will automatically download yolov8n.pt if you don't have it
    model = YOLO("yolov8n.pt")

    print("Exporting to TensorRT FP16 Engine (This may take a few minutes)...")
    # format="engine" automatically exports to ONNX first, then builds the TRT Engine
    # half=True uses FP16 for massive speed gains
    # imgsz=640 is standard YOLO resolution
    model.export(format="engine", half=True, imgsz=640, workspace=4)
    
    print("✅ Export Complete! Engine saved as 'yolov8n.engine'")

if __name__ == "__main__":
    export_to_tensorrt()

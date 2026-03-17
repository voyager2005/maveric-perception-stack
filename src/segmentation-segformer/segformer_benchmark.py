import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import time

# --- Configuration ---
# We use B5 because that is what you wanted, but we can swap to B0/B1 if B5 is too slow
MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
DEVICE = "cuda"
NUM_WARMUP = 10
NUM_TEST_LOOPS = 50

def benchmark():
    print(f"Initializing {MODEL_ID} on {torch.cuda.get_device_name(0)}...")
    
    # 1. Load Model
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    # 2. Create Dummy Input (1024x2048 is standard Cityscapes resolution)
    # We use a dummy image to measure COMPUTE limits, not Disk I/O limits
    dummy_image = Image.new('RGB', (2048, 1024), color = 'red')

    # 3. Setup CUDA Events for precise timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # --- Warm Up ---
    # GPU needs to compile kernels; first few runs are always slow
    print(f"Warming up GPU with {NUM_WARMUP} passes...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            inputs = processor(images=dummy_image, return_tensors="pt").to(DEVICE)
            _ = model(**inputs)
    
    # --- Benchmarking ---
    print(f"Running benchmark ({NUM_TEST_LOOPS} iterations)...")
    timings = []

    with torch.no_grad():
        for i in range(NUM_TEST_LOOPS):
            # Start Timer
            starter.record()
            
            # A. Pre-process (CPU -> GPU move)
            inputs = processor(images=dummy_image, return_tensors="pt").to(DEVICE)
            
            # B. Inference (The heavy lifting)
            outputs = model(**inputs)
            logits = outputs.logits
            
            # C. Post-process (Downstream Data Retrieval)
            # Resize mask to original 1024x2048 so it matches LiDAR projection
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(1024, 2048),
                mode='bilinear',
                align_corners=False
            )
            # Extract Class IDs (The actual data 'PointPainting' needs)
            # We move to CPU because the next node (Fusion) might run on CPU or need synchronization
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu()
            
            # Stop Timer
            ender.record()
            
            # Wait for GPU to finish
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # returns milliseconds
            timings.append(curr_time)

    # --- Results ---
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000 / avg_ms

    print("-" * 40)
    print(f"Results for SegFormer B5 on {torch.cuda.get_device_name(0)}")
    print("-" * 40)
    print(f"Avg Latency: {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"Throughput:  {fps:.2f} FPS")
    print("-" * 40)
    
    if fps < 10:
        print("⚠ WARNING: Latency is too high for high-speed autonomous driving.")
        print("Suggestion: Try SegFormer B2 or B1, or use TensorRT optimization.")
    else:
        print("✅ STATUS: Performance looks viable for real-time perception.")

if __name__ == "__main__":
    benchmark()

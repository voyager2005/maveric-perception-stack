import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
ENGINE_PATH = "segformer_b1_python_build.engine"
IMAGE_DIR = "/home/cv/Documents/SegTransformer/Data/Images/" # Path to your extracted frames
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Cityscapes Palette (Standard for visualization)
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
], dtype=np.uint8)

class SegFormerTRT:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                self.input_shape = shape
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                self.output_shape = shape

    def infer(self, image):
        # 1. Resize & Normalize
        img_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_norm = (img_resized.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        img_chw = img_norm.transpose(2, 0, 1)
        np.copyto(self.inputs[0]['host'], np.ascontiguousarray(img_chw.ravel()))

        # 2. Inference
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        for i in range(len(self.inputs)):
             self.context.set_tensor_address(self.engine.get_tensor_name(i), int(self.inputs[i]['device']))
        for i in range(len(self.outputs)):
             idx = len(self.inputs) + i
             self.context.set_tensor_address(self.engine.get_tensor_name(idx), int(self.outputs[i]['device']))
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'].reshape(self.output_shape)

def main():
    # 1. Load Engine
    if not os.path.exists(ENGINE_PATH):
        print("Engine file missing!")
        return
    model = SegFormerTRT(ENGINE_PATH)
    print("Engine Loaded.")

    # 2. Get a real image
    images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return
    
    # Pick the 10th image (or first if fewer) to avoid boring startup frames
    img_path = images[67] if len(images) > 10 else images[0]
    print(f"Testing on: {img_path}")
    
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to read image.")
        return

    # 3. Run Inference
    logits = model.infer(image) # Shape: (1, 19, 256, 256)

    # 4. Process Output
    # Get Class IDs (argmax)
    pred_mask = np.argmax(logits, axis=1)[0] # Shape: (256, 256)
    
    # Resize mask back to original image size for display
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Colorize
    color_mask = PALETTE[pred_mask_resized]

    # 5. Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Show Original (Convert BGR to RGB for Matplotlib)
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Frame")
    axs[0].axis('off')

    # Show Segmentation
    axs[1].imshow(color_mask)
    axs[1].set_title(f"SegFormer B1 (TensorRT) - {54.78} FPS")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    print("Sanity check complete. Close the plot window to finish.")

if __name__ == "__main__":
    main()

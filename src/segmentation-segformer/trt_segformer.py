import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os

# Suppress TRT warnings
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class SegFormerTRT:
    def __init__(self, engine_path):
        print(f"Loading TensorRT Engine: {engine_path}")
        
        # 1. Load the Engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 2. Allocate Memory (Host vs Device)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            # Get size and dtype
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            size = trt.volume(shape)
            
            # Allocate Page-Locked Memory (Faster for GPU transfers)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to list of bindings
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                self.input_shape = shape
                print(f"Input Binding: {shape}")
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                self.output_shape = shape
                print(f"Output Binding: {shape}")

    def preprocess(self, image):
        # Resize to Model Input (1024x1024)
        # Note: If your camera is 1920x1080, this squashing is okay for segmentation
        img_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        
        # Normalize (Standard Cityscapes mean/std)
        mean = np.array([123.675, 116.28, 103.53]).astype(np.float32)
        std = np.array([58.395, 57.12, 57.375]).astype(np.float32)
        
        img_float = img_resized.astype(np.float32)
        img_norm = (img_float - mean) / std
        
        # HWC -> CHW (Channels First)
        img_chw = img_norm.transpose(2, 0, 1)
        
        # Add Batch Dimension (1, 3, 1024, 1024) and Flatten
        return np.ascontiguousarray(img_chw.ravel())

    def infer(self, image):
        # A. Preprocess (CPU)
        processed_img = self.preprocess(image)
        np.copyto(self.inputs[0]['host'], processed_img)
        
        # B. Inference (GPU)
        start_time = time.time()
        
        # 1. Host -> Device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 2. Execute
        # Set tensor address for all bindings
        for i in range(len(self.inputs)):
             self.context.set_tensor_address(self.engine.get_tensor_name(i), int(self.inputs[i]['device']))
        for i in range(len(self.outputs)):
             idx = len(self.inputs) + i
             self.context.set_tensor_address(self.engine.get_tensor_name(idx), int(self.outputs[i]['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 3. Device -> Host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # 4. Synchronize
        self.stream.synchronize()
        end_time = time.time()
        
        # C. Postprocess
        # Reshape output to (Batch, Classes, Height/4, Width/4)
        output = self.outputs[0]['host'].reshape(self.output_shape)
        
        return output, (end_time - start_time) * 1000

# --- Test Block ---
if __name__ == "__main__":
    # UPDATED FILENAME
    ENGINE_FILE = "segformer_b1_python_build.engine" 
    
    if not os.path.exists(ENGINE_FILE):
        print("Engine not found! Run build_engine.py first.")
        exit()

    trt_model = SegFormerTRT(ENGINE_FILE)
    
    # Load Dummy Image
    dummy_img = cv2.imread("testimage.png") # Or use np.zeros if file missing
    if dummy_img is None:
        dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)

    print("Running Warmup...")
    for _ in range(10):
        trt_model.infer(dummy_img)

    print("Running Benchmark (Python Overhead Included)...")
    latencies = []
    for _ in range(100):
        _, ms = trt_model.infer(dummy_img)
        latencies.append(ms)
        
    avg_lat = np.mean(latencies)
    print(f"Average Python Inference Time: {avg_lat:.2f} ms ({1000/avg_lat:.2f} FPS)")

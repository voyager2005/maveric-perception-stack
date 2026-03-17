import tensorrt as trt
import os

# 1. Setup Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path, engine_file_path):
    print(f"Building engine from {onnx_file_path}...")
    
    # 2. Initialize Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 3. Parse ONNX
    if not os.path.exists(onnx_file_path):
        print(f"Error: {onnx_file_path} not found.")
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 4. Config Optimization (FP16)
    if builder.platform_has_fast_fp16:
        print("FP16 is supported. Enabling...")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Memory Pool (Give it enough RAM to work)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024) # 2GB

    # 5. Build Serialized Engine
    print("Building... (This may take 1-3 minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Success! Engine saved to {engine_file_path}")
    else:
        print("Build failed.")

if __name__ == "__main__":
    # Make sure these filenames match yours
    ONNX_PATH = "segformer_b1.onnx" 
    ENGINE_PATH = "segformer_b1_python_build.engine"
    
    build_engine(ONNX_PATH, ENGINE_PATH)

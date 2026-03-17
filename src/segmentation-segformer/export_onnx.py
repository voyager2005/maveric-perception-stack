import torch
from transformers import SegformerForSemanticSegmentation

# 1. Config
MODEL_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
ONNX_PATH = "segformer_b1.onnx"

print(f"Loading {MODEL_ID}...")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()
model.cuda()

# 2. Create Dummy Input
# Shape: (Batch, Channels, Height, Width)
# Cityscapes standard is usually 1024x2048. If you want faster speed, reduce this to 512x1024.
# Let's try 512x1024 for speed first, or keep 1024x2048 if you need long-range detection.
# We will start with 1024x1024 (square) to match the model training config.
dummy_input = torch.randn(1, 3, 1024, 1024, device='cuda')

# 3. Export
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=13,
    input_names=['input'],
    output_names=['logits'],
    do_constant_folding=True,
    dynamic_axes=None  # We want STATIC axes for maximum TRT performance
)

print(f"Success! Model exported to {ONNX_PATH}")

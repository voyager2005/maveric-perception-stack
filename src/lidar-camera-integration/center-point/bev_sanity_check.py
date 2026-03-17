import numpy as np
import cv2
import os
from voxelizer import NumpyCenterPointVoxelizer

# Cityscapes 19-class color palette (BGR for OpenCV)
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

def visualize_bev_semantics():
    # 1. Load the first available painted sweep
    test_bin_path = "../lidar_integration/sweep_lidar_integration"
    sample_file = [f for f in os.listdir(test_bin_path) if f.endswith('.bin')][0]
    file_path = os.path.join(test_bin_path, sample_file)
    
    print(f"Loading {sample_file}...")
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 24)
    
    # 2. Voxelize
    voxelizer = NumpyCenterPointVoxelizer()
    voxels, coords, _ = voxelizer.generate(points)
    
    # 3. Create an empty 1024x1024 RGB image (Black background)
    grid_size = voxelizer.grid_size # [1024, 1024, 40]
    bev_img = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
    
    # 4. Extract classes from the one-hot vectors (dims 5 to 23)
    # Sum the class probabilities across all points in a single voxel
    class_sums = np.sum(voxels[:, :, 5:24], axis=1) # Shape: (14968, 19)
    
    # Find the most prominent class in each voxel
    dominant_classes = np.argmax(class_sums, axis=1) # Shape: (14968,)
    
    # Check if a voxel has zero painted features (meaning it was outside the 3 front cameras)
    max_sums = np.max(class_sums, axis=1)
    painted_mask = max_sums > 0
    unpainted_mask = max_sums == 0
    
    # 5. Map to coordinates
    y = coords[:, 1] # Y maps to image rows
    x = coords[:, 2] # X maps to image columns
    
    # Draw unpainted raw LiDAR as faint white dots
    bev_img[y[unpainted_mask], x[unpainted_mask]] = [100, 100, 100]
    
    # Draw painted LiDAR with Cityscapes colors
    bev_img[y[painted_mask], x[painted_mask]] = CITYSCAPES_COLORS[dominant_classes[painted_mask]]
    
    # 6. Draw Ego Vehicle in the center
    center_y, center_x = grid_size[1] // 2, grid_size[0] // 2
    cv2.circle(bev_img, (center_x, center_y), 5, (0, 0, 255), -1) # Red dot for ego
    
    # Note: Lidar Y-axis might be inverted depending on viewer coordinate systems, 
    # flipping it vertically usually makes it align visually with what we expect (forward is up)
    bev_img = cv2.flip(bev_img, 0)
    
    # 7. Save
    output_name = "bev_semantic_sanity_check.png"
    cv2.imwrite(output_name, bev_img)
    print(f"\n✅ BEV Image saved as: {output_name}")
    print("Open this file. You should see a top-down view of the road, cars, and unpainted lidar surrounding the red ego-vehicle!")

if __name__ == "__main__":
    visualize_bev_semantics()

import numpy as np
import os

# ==========================================
# 🛑 CENTERPOINT VOXEL CONFIGURATION
# ==========================================
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.1, 0.1, 0.2]
MAX_POINTS_PER_VOXEL = 10 
MAX_VOXELS_TRAIN = 120000 
NUM_POINT_FEATURES = 24
# ==========================================

class NumpyCenterPointVoxelizer:
    def __init__(self):
        print("Initializing Native NumPy Voxel Generator...")
        self.pc_range = np.array(POINT_CLOUD_RANGE, dtype=np.float32)
        self.voxel_size = np.array(VOXEL_SIZE, dtype=np.float32)
        
        # Grid Size = (Max - Min) / Voxel Size
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[0:3]) / self.voxel_size).astype(np.int32)
        print(f"✅ Voxel Grid Size (X, Y, Z): {self.grid_size}")

    def generate(self, points):
        """
        Numpy implementation mimicking spconv's PointToVoxel
        """
        # 1. Filter points outside the point cloud range
        mask = (points[:, 0] >= self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) & \
               (points[:, 1] >= self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) & \
               (points[:, 2] >= self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
        points = points[mask]

        # 2. Convert raw point coordinates to voxel grid indices
        # Shift coordinates so minimum is 0, then divide by voxel size
        shifted_coords = points[:, :3] - self.pc_range[0:3]
        voxel_coords = np.floor(shifted_coords / self.voxel_size).astype(np.int32)
        
        # CenterPoint expects coordinates in (Z, Y, X) order
        voxel_coords = voxel_coords[:, ::-1]

        # 3. Find unique voxels and map points to them
        # We use a trick: hash the 3D coordinates into a 1D integer to find unique voxels quickly
        hash_multiplier = np.array([self.grid_size[1] * self.grid_size[0], self.grid_size[0], 1], dtype=np.int64)
        voxel_hashes = np.sum(voxel_coords * hash_multiplier, axis=1)
        
        # Get unique hashes and the inverse mapping
        unique_hashes, inverse_indices = np.unique(voxel_hashes, return_inverse=True)
        
        num_voxels = min(len(unique_hashes), MAX_VOXELS_TRAIN)
        
        # Initialize output arrays
        voxels = np.zeros((num_voxels, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES), dtype=np.float32)
        coords = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)

        # Populate the voxels
        for i in range(len(points)):
            voxel_idx = inverse_indices[i]
            
            # Stop if we hit the max total voxels limit
            if voxel_idx >= num_voxels:
                continue
                
            point_idx = num_points_per_voxel[voxel_idx]
            
            # Stop adding points to this specific voxel if it's full
            if point_idx < MAX_POINTS_PER_VOXEL:
                voxels[voxel_idx, point_idx] = points[i]
                coords[voxel_idx] = voxel_coords[i]
                num_points_per_voxel[voxel_idx] += 1

        return voxels, coords, num_points_per_voxel

if __name__ == "__main__":
    test_bin_path = "../lidar_integration/sweep_lidar_integration"
    
    try:
        sample_file = [f for f in os.listdir(test_bin_path) if f.endswith('.bin')][0]
        file_path = os.path.join(test_bin_path, sample_file)
        
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 24)
        print(f"Loaded points shape: {points.shape}")
        
        voxelizer = NumpyCenterPointVoxelizer()
        voxels, coords, num_pts = voxelizer.generate(points)
        
        print(f"\nResults:")
        print(f"Voxels Shape: {voxels.shape}")
        print(f"Coordinates Shape: {coords.shape}")
        print(f"Num Points per Voxel Shape: {num_pts.shape}")
        
    except IndexError:
         print("No .bin files found in the sweep directory.")

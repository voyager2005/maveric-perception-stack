import torch
import torch.nn as nn

class VoxelFeatureEncoder(nn.Module):
    def __init__(self, num_input_features=24, num_filters=64):
        super().__init__()
        # A simple linear layer to expand our 24 painted features
        self.linear = nn.Linear(num_input_features, num_filters)
        self.batch_norm = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, voxels, num_points):
        """
        voxels: (V, P, C) -> e.g., (14968, 10, 24)
        num_points: (V,) -> e.g., (14968,)
        """
        V, P, C = voxels.shape
        
        # 1. Flatten to pass through the linear layer
        # Shape becomes (V * P, C) -> (149680, 24)
        x = voxels.view(-1, C)
        
        # 2. Extract Features
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        # 3. Reshape back to voxels
        # Shape becomes (V, P, num_filters) -> (14968, 10, 64)
        x = x.view(V, P, -1)
        
        # 4. Max Pooling over the points in each voxel (dim=1)
        # This grabs the most prominent features (like high probability classes)
        # Output shape: (V, 64) -> (14968, 64)
        voxel_features, _ = torch.max(x, dim=1)
        
        return voxel_features

class BEVScatter(nn.Module):
    def __init__(self, grid_size_x=1024, grid_size_y=1024):
        super().__init__()
        self.nx = grid_size_x
        self.ny = grid_size_y

    def forward(self, voxel_features, coords):
        """
        voxel_features: (V, 64)
        coords: (V, 3) where columns are (Z, Y, X)
        """
        # Collapse the Z axis by ignoring coords[:, 0] and just using Y and X
        # For PointPillars, we just scatter to a 2D plane.
        
        batch_size = 1 # We are processing 1 sweep at a time right now
        num_filters = voxel_features.shape[-1]
        
        # Initialize an empty dense BEV canvas: (Batch, Channels, Y, X)
        # Shape: (1, 64, 1024, 1024)
        canvas = torch.zeros(
            batch_size, num_filters, self.ny, self.nx, 
            dtype=voxel_features.dtype, 
            device=voxel_features.device
        )
        
        # Extract X and Y coordinates
        y_indices = coords[:, 1].long()
        x_indices = coords[:, 2].long()
        
        # Scatter the features onto the 2D canvas
        # Note: If multiple voxels land on the same X,Y (different Z), this overwrites.
        # In a full PointPillars model, Z is concatenated into the channels before this step.
        canvas[0, :, y_indices, x_indices] = voxel_features.t()
        
        return canvas

# --- Test Block ---
if __name__ == "__main__":
    # Simulate the outputs from your NumPy voxelizer
    V = 14968
    dummy_voxels = torch.rand(V, 10, 24)
    dummy_num_points = torch.randint(1, 11, (V,))
    
    # Create random coordinates inside our 1024x1024 grid
    dummy_coords = torch.zeros(V, 3)
    dummy_coords[:, 0] = torch.randint(0, 40, (V,)) # Z
    dummy_coords[:, 1] = torch.randint(0, 1024, (V,)) # Y
    dummy_coords[:, 2] = torch.randint(0, 1024, (V,)) # X

    print("Running Voxel Feature Encoder...")
    vfe = VoxelFeatureEncoder()
    encoded_features = vfe(dummy_voxels, dummy_num_points)
    print(f"Encoded Features Shape: {encoded_features.shape}") # Should be (14968, 64)

    print("\nRunning BEV Scatter...")
    scatter = BEVScatter()
    bev_image = scatter(encoded_features, dummy_coords)
    print(f"Dense BEV Image Shape: {bev_image.shape}") # Should be (1, 64, 1024, 1024)

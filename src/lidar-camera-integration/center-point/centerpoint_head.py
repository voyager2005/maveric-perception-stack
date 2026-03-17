import torch
import torch.nn as nn

class BEVBackbone2D(nn.Module):
    """
    A lightweight 2D CNN backbone to extract features from the BEV pseudo-image.
    It downsamples the grid to increase the receptive field, then upsamples 
    to fuse multi-scale features.
    """
    def __init__(self, in_channels=64, out_channels=256):
        super().__init__()
        
        # Block 1: Downsample 1024x1024 -> 512x512
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: Downsample 512x512 -> 256x256
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Upsample blocks to bring both back to 512x512 for dense prediction
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (Batch, 64, 1024, 1024)
        out1 = self.block1(x)       # Shape: (Batch, 128, 512, 512)
        out2 = self.block2(out1)    # Shape: (Batch, 256, 256, 256)
        
        up1 = self.deconv1(out1)    # Shape: (Batch, 128, 512, 512)
        up2 = self.deconv2(out2)    # Shape: (Batch, 128, 512, 512)
        
        # Concatenate multi-scale features along the channel dimension
        # Output Shape: (Batch, 256, 512, 512)
        return torch.cat([up1, up2], dim=1)


class CenterHead(nn.Module):
    """
    The multi-task detection head. It splits the backbone features into 
    separate convolutional paths for heatmaps, sizes, rotations, etc.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        
        # We group classes in ADAS pipelines. E.g., 0: Vehicle, 1: Pedestrian, 2: Cyclist
        self.num_classes = num_classes
        
        # Shared convolution configuration for each regression head
        head_conv = 64
        
        def make_head(out_dim):
            return nn.Sequential(
                nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_dim, kernel_size=1)
            )

        # 1. Heatmap: Probability of an object center existing at this pixel
        self.heatmap_head = make_head(num_classes)
        
        # 2. Sub-pixel Offset: Since we downsampled to 512x512, an object's true center 
        # might fall between pixels. This predicts that fine (x, y) continuous offset.
        self.offset_head = make_head(2)
        
        # 3. Z Elevation: The absolute height of the object's center in meters
        self.z_head = make_head(1)
        
        # 4. 3D Dimensions: Width, Length, Height (usually log-encoded during training)
        self.dim_head = make_head(3)
        
        # 5. Rotation: sin(yaw) and cos(yaw) for continuous angle representation
        self.rot_head = make_head(2)
        
        # 6. Velocity: v_x and v_y in meters per second (Crucial for AB3DMOT)
        self.vel_head = make_head(2)

    def forward(self, x):
        # x shape: (Batch, 256, 512, 512)
        preds_dict = {
            "heatmap": torch.sigmoid(self.heatmap_head(x)), # Sigmoid for probabilities [0, 1]
            "offset": self.offset_head(x),
            "z": self.z_head(x),
            "dim": self.dim_head(x),
            "rot": self.rot_head(x),
            "vel": self.vel_head(x)
        }
        return preds_dict


# --- Test Block ---
if __name__ == "__main__":
    # Simulate the dense BEV output from our previous scatter step
    # Batch=1, Channels=64, Y=1024, X=1024
    dummy_bev = torch.rand(1, 64, 1024, 1024)
    print(f"Input BEV Shape: {dummy_bev.shape}")

    print("\nRunning BEV Backbone...")
    backbone = BEVBackbone2D()
    backbone_features = backbone(dummy_bev)
    print(f"Backbone Output Shape: {backbone_features.shape}")

    print("\nRunning CenterPoint Head...")
    head = CenterHead()
    predictions = head(backbone_features)

    print("\nPrediction Tensor Shapes (Grid Resolution 512x512):")
    for key, tensor in predictions.items():
        print(f" - {key.ljust(10)}: {tensor.shape}")

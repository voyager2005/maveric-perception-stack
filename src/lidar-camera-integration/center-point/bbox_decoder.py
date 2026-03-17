import torch
import torch.nn.functional as F

class CenterPointDecoder:
    def __init__(self, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], voxel_size=[0.1, 0.1, 0.2], feature_map_stride=2):
        """
        pc_range: [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
        voxel_size: Original size of voxels before backbone
        feature_map_stride: How much the backbone downsampled the grid (e.g., 1024 -> 512 means stride=2)
        """
        self.pc_range = pc_range
        
        # Calculate the real-world metric size of a single pixel on the feature map
        self.out_size_factor = feature_map_stride
        self.feature_x_size = voxel_size[0] * self.out_size_factor
        self.feature_y_size = voxel_size[1] * self.out_size_factor

    def _nms_fast(self, heatmaps):
        """
        Acts as a fast, anchor-free Non-Maximum Suppression.
        Keeps only the local peaks in a 3x3 neighborhood.
        """
        # Apply 3x3 max pooling with stride 1 and padding 1 to keep tensor size the same
        hmax = F.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)
        # Only keep pixels that are equal to their neighborhood maximum
        keep_mask = (hmax == heatmaps).float()
        return heatmaps * keep_mask

    def decode(self, preds_dict, score_threshold=0.3, max_objects=100):
        """
        Converts the raw feature map dictionaries into a list of 3D bounding boxes.
        """
        heatmap = preds_dict['heatmap']
        batch_size, num_classes, H, W = heatmap.shape
        
        # 1. Fast NMS via Max Pooling
        heatmap = self._nms_fast(heatmap)
        
        # 2. Flatten the heatmap to find the top scoring indices across the whole grid
        # Shape: (Batch, Classes * H * W)
        heatmap_flat = heatmap.view(batch_size, -1)
        
        # Get the top N highest scoring pixels
        top_scores, top_indices = torch.topk(heatmap_flat, max_objects, dim=1)
        
        # 3. Decode 1D indices back to Class, Y, and X coordinates
        top_classes = (top_indices // (H * W)).long()
        top_indices_2d = top_indices % (H * W)
        top_ys = (top_indices_2d // W).long()
        top_xs = (top_indices_2d % W).long()

        boxes_3d = []
        
        # Process the single batch element (we are assuming batch_size=1 for streaming inference)
        b = 0 
        for i in range(max_objects):
            score = top_scores[b, i].item()
            if score < score_threshold:
                continue # Skip low-confidence detections
                
            y_grid = top_ys[b, i].item()
            x_grid = top_xs[b, i].item()
            class_id = top_classes[b, i].item()
            
            # Extract attributes from the regression maps at this specific (y, x) location
            offset_x = preds_dict['offset'][b, 0, y_grid, x_grid].item()
            offset_y = preds_dict['offset'][b, 1, y_grid, x_grid].item()
            
            z = preds_dict['z'][b, 0, y_grid, x_grid].item()
            
            # Dimensions are typically predicted as log(dim) to prevent negative sizes, 
            # so we use torch.exp() to get real meters.
            w = torch.exp(preds_dict['dim'][b, 0, y_grid, x_grid]).item()
            l = torch.exp(preds_dict['dim'][b, 1, y_grid, x_grid]).item()
            h = torch.exp(preds_dict['dim'][b, 2, y_grid, x_grid]).item()
            
            sin_theta = preds_dict['rot'][b, 0, y_grid, x_grid].item()
            cos_theta = preds_dict['rot'][b, 1, y_grid, x_grid].item()
            yaw = torch.atan2(torch.tensor(sin_theta), torch.tensor(cos_theta)).item()
            
            v_x = preds_dict['vel'][b, 0, y_grid, x_grid].item()
            v_y = preds_dict['vel'][b, 1, y_grid, x_grid].item()
            
            # Convert Grid Coordinates to Real-World Metric Coordinates
            x_metric = (x_grid + offset_x) * self.feature_x_size + self.pc_range[0]
            y_metric = (y_grid + offset_y) * self.feature_y_size + self.pc_range[1]
            
            # Package into a clean dictionary
            box = {
                "class_id": class_id,
                "score": score,
                "x": x_metric,
                "y": y_metric,
                "z": z,
                "w": w,
                "l": l,
                "h": h,
                "yaw": yaw,
                "vx": v_x,
                "vy": v_y
            }
            boxes_3d.append(box)
            
        return boxes_3d

# --- Test Block ---
if __name__ == "__main__":
    from centerpoint_head import BEVBackbone2D, CenterHead
    
    # 1. Create Dummy BEV Input and run it through the network
    dummy_bev = torch.rand(1, 64, 1024, 1024)
    backbone = BEVBackbone2D()
    head = CenterHead(num_classes=3)
    
    # Evaluate mode to disable batch norm randomness during testing
    backbone.eval()
    head.eval()
    
    with torch.no_grad():
        backbone_features = backbone(dummy_bev)
        predictions = head(backbone_features)

    # 2. Inject a fake "perfect" prediction into the dummy tensors to test the decoder
    # Let's say a vehicle (class 0) is at grid location y=250, x=250
    predictions['heatmap'][0, 0, 250, 250] = 0.95 
    predictions['offset'][0, :, 250, 250] = torch.tensor([0.5, 0.5]) # Half a pixel offset
    predictions['dim'][0, :, 250, 250] = torch.log(torch.tensor([1.8, 4.5, 1.5])) # Standard car size
    predictions['vel'][0, :, 250, 250] = torch.tensor([12.0, 0.0]) # Moving forward at 12 m/s

    # 3. Decode
    print("Decoding Predictions...")
    decoder = CenterPointDecoder()
    detected_boxes = decoder.decode(predictions, score_threshold=0.5)
    
    print(f"\nFound {len(detected_boxes)} valid objects above threshold.")
    if len(detected_boxes) > 0:
        print("Top Detection:")
        for k, v in detected_boxes[0].items():
            if isinstance(v, float):
                print(f"  {k.ljust(8)}: {v:.3f}")
            else:
                print(f"  {k.ljust(8)}: {v}")

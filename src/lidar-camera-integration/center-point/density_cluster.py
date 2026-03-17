import os
import numpy as np
import cv2
import glob
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini"
PAINTED_DIR = os.path.join(DATA_DIR, "painted_lidar")
OUTPUT_DIR = "clustering_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Volume Approximations & Thresholds (Length, Width, Height, Min Points)
CLASS_PRIORS = {
    "car":        {"l": 4.5, "w": 1.9, "h": 1.5, "min_pts": 15},
    "truck":      {"l": 8.0, "w": 2.5, "h": 3.0, "min_pts": 25},
    "bus":        {"l": 10.0, "w": 2.8, "h": 3.5, "min_pts": 30},
    "person":     {"l": 0.6, "w": 0.6, "h": 1.7, "min_pts": 5},
    "motorcycle": {"l": 2.0, "w": 0.8, "h": 1.5, "min_pts": 8},
    "bicycle":    {"l": 1.8, "w": 0.6, "h": 1.5, "min_pts": 8}
}

# 19-class indices mapped to our dynamic classes
DYNAMIC_CLASSES = {
    11: "person", 13: "car", 14: "truck", 
    15: "bus", 17: "motorcycle", 18: "bicycle"
}

# BEV Visualization Params
BEV_RES = 0.1 # 10cm per pixel
BEV_SIZE = 1024 # 100m x 100m canvas
# ==========================================

def fit_3d_bbox_with_priors(cluster_points, class_name):
    """
    Fits an OBB using PCA, and applies Class Priors if the LiDAR 
    only captured a partial surface of the object.
    """
    center = np.mean(cluster_points, axis=0)
    priors = CLASS_PRIORS[class_name]
    
    # Estimate Orientation using PCA on X-Y
    pca = PCA(n_components=2)
    pca.fit(cluster_points[:, :2])
    
    heading_vec = pca.components_[0]
    yaw = np.arctan2(heading_vec[1], heading_vec[0])
    
    # Rotate points to local frame to find raw dimensions
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    local_pts = np.dot(cluster_points[:, :2] - center[:2], rot_mat.T)
    
    raw_l = np.max(local_pts[:, 0]) - np.min(local_pts[:, 0])
    raw_w = np.max(local_pts[:, 1]) - np.min(local_pts[:, 1])
    raw_h = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
    
    # --- HEURISTIC: VOLUME APPROXIMATION ---
    # If the measured length is less than 60% of a standard car, 
    # it means we only hit one side of it. We snap to the prior.
    final_l = priors["l"] if raw_l < (priors["l"] * 0.6) else (raw_l + 0.2)
    final_w = priors["w"] if raw_w < (priors["w"] * 0.6) else (raw_w + 0.2)
    final_h = priors["h"] if raw_h < (priors["h"] * 0.6) else raw_h
    
    return [center[0], center[1], center[2], final_l, final_w, final_h, yaw]

def draw_cluster_bev(points_3d, objects, filename):
    """Draws the point cloud and bounding boxes, saving to OUTPUT_DIR"""
    img = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)
    cx, cy = BEV_SIZE // 2, BEV_SIZE // 2
    
    # 1. Draw Ego
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    
    # 2. Draw Points (Faint grey)
    img_x = cx - (points_3d[:, 1] / BEV_RES).astype(int)
    img_y = cy - (points_3d[:, 0] / BEV_RES).astype(int)
    valid_mask = (img_x >= 0) & (img_x < BEV_SIZE) & (img_y >= 0) & (img_y < BEV_SIZE)
    img[img_y[valid_mask], img_x[valid_mask]] = [80, 80, 80]
    
    # 3. Draw Approximated Bounding Boxes
    for obj in objects:
        x, y, z, l, w, h, yaw = obj['bbox']
        
        # Convert to pixels
        px = int(cx - (y / BEV_RES))
        py = int(cy - (x / BEV_RES))
        pl = int(l / BEV_RES)
        pw = int(w / BEV_RES)
        
        # Draw Rotated Box
        rect = ((px, py), (pw, pl), np.degrees(-yaw))
        box_points = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)
        
        # Add Label
        label = f"{obj['class']}"
        cv2.putText(img, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)

def process_painted_sweep(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 24)
    points_3d = data[:, :3]
    semantics = data[:, 5:] 
    class_ids = np.argmax(semantics, axis=1)
    
    all_objects = []

    for class_idx, class_name in DYNAMIC_CLASSES.items():
        mask = (class_ids == class_idx)
        target_points = points_3d[mask]
        
        min_required_pts = CLASS_PRIORS[class_name]["min_pts"]
        if len(target_points) < min_required_pts:
            continue
            
        # DBSCAN: Separate instances
        clustering = DBSCAN(eps=0.8, min_samples=min_required_pts).fit(target_points)
        labels = clustering.labels_
        
        for label in set(labels):
            if label == -1: continue # Skip noise
            
            cluster_points = target_points[labels == label]
            
            # Enforce minimum points again per cluster
            if len(cluster_points) < min_required_pts:
                continue
                
            bbox = fit_3d_bbox_with_priors(cluster_points, class_name)
            all_objects.append({
                "class": class_name,
                "bbox": bbox, 
                "num_points": len(cluster_points)
            })
            
    return points_3d, all_objects

def run_pipeline():
    bin_files = sorted(glob.glob(os.path.join(PAINTED_DIR, "*.bin")))
    if not bin_files:
        print(f"❌ No painted .bin files found in {PAINTED_DIR}")
        return

    print(f"Starting Semantic Clustering on {len(bin_files)} files...")
    
    for i, bin_path in enumerate(bin_files[:10]): # Process first 10 for quick testing
        filename = os.path.basename(bin_path)
        img_filename = filename.replace("_painted.bin", "_cluster.jpg")
        
        # Run clustering & approximation
        points_3d, objects = process_painted_sweep(bin_path)
        
        # Save BEV Visualization
        draw_cluster_bev(points_3d, objects, img_filename)
        
        print(f"\n--- Frame {i}: {filename} ---")
        for obj in objects:
            x, y, z, l, w, h, yaw = obj['bbox']
            print(f"  [{obj['class']}] Pos:({x:>5.1f}, {y:>5.1f}) | Dim:({l:.1f}x{w:.1f}x{h:.1f}) | Pts:{obj['num_points']}")
            
    print(f"\n✅ Visualizations saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_pipeline()

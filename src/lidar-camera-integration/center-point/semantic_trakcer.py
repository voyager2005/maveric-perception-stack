import os
import numpy as np
import glob
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from tracker import AB3DMOT # The tracker we wrote earlier!

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
INPUT_DIR = "../lidar_integration/sweep_lidar_integration"
# Cityscapes dynamic classes we care about tracking
# 11: person, 12: rider, 13: car, 14: truck, 15: bus, 17: motorcycle, 18: bicycle
TRACKED_CLASSES = [11, 12, 13, 14, 15, 17, 18] 

# DBSCAN grouping distance in meters. (e.g., points within 1.0m belong to the same car)
CLUSTER_DISTANCE = 1.0 
MIN_POINTS_PER_CLUSTER = 10 # Throw away ghost artifacts
# ==========================================

def extract_bounding_boxes(points, classes):
    """
    Takes N x 3 points and their class IDs, clusters them into instances, 
    and fits 3D oriented bounding boxes.
    """
    detected_boxes = []
    
    # Process each class separately (so a person standing next to a car doesn't merge into it)
    for target_class in np.unique(classes):
        if target_class not in TRACKED_CLASSES:
            continue
            
        class_mask = classes == target_class
        class_points = points[class_mask]
        
        if len(class_points) < MIN_POINTS_PER_CLUSTER:
            continue
            
        # 1. Cluster the points using physical density
        clustering = DBSCAN(eps=CLUSTER_DISTANCE, min_samples=MIN_POINTS_PER_CLUSTER).fit(class_points)
        labels = clustering.labels_
        
        # 2. Fit a bounding box to each valid cluster
        for label in np.unique(labels):
            if label == -1: continue # -1 is DBSCAN's label for random noise points
            
            cluster_pts = class_points[labels == label]
            
            # Center coordinates
            center = np.mean(cluster_pts, axis=0)
            
            # Elevation and Height
            z_min, z_max = np.min(cluster_pts[:, 2]), np.max(cluster_pts[:, 2])
            h = z_max - z_min
            z = center[2]
            
            # PCA to find the Yaw (Orientation) on the ground plane (X, Y)
            pca = PCA(n_components=2)
            pca.fit(cluster_pts[:, :2])
            
            # The primary eigenvector gives us the heading
            heading_vector = pca.components_[0]
            yaw = np.arctan2(heading_vector[1], heading_vector[0])
            
            # Project points onto the PCA axes to get precise Length and Width
            transformed_pts = pca.transform(cluster_pts[:, :2])
            l = np.max(transformed_pts[:, 0]) - np.min(transformed_pts[:, 0])
            w = np.max(transformed_pts[:, 1]) - np.min(transformed_pts[:, 1])
            
            # Add padding since LiDAR rarely hits the absolute extreme edges of a car
            l += 0.2
            w += 0.2
            
            detected_boxes.append({
                "class_id": target_class,
                "score": 1.0, # Semantic clusters have 100% implicit confidence from Segformer
                "x": center[0], "y": center[1], "z": z,
                "l": l, "w": w, "h": h,
                "yaw": yaw,
                "vx": 0.0, "vy": 0.0 # Initialized at 0, Kalman Filter will update this
            })
            
    return detected_boxes

def run_tracking_pipeline():
    bin_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.bin")))
    if not bin_files:
        print("No painted .bin files found!")
        return

    # Initialize our AB3DMOT tracker (max_age=3 frames before dropping a lost track)
    tracker = AB3DMOT(max_age=3, min_hits=1, distance_threshold=3.0)
    
    # Assuming NuScenes 10Hz LiDAR, dt = 0.1s
    dt = 0.1 

    print(f"Starting Semantic Clustering & Tracking on {len(bin_files)} frames...\n")
    
    for i, file_path in enumerate(bin_files):
        # 1. Load Painted Point Cloud (N x 24)
        painted_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 24)
        
        # 2. Extract XYZ coordinates and Semantic Classes
        points_3d = painted_cloud[:, :3]
        
        # Classes are stored as one-hot vectors in dimensions 5 through 23
        one_hot_classes = painted_cloud[:, 5:24]
        
        # If a point has no painted class (outside camera FOV), its sum is 0
        painted_mask = np.sum(one_hot_classes, axis=1) > 0
        
        valid_points = points_3d[painted_mask]
        valid_classes = np.argmax(one_hot_classes[painted_mask], axis=1)
        
        # 3. Extract purely mathematical bounding boxes
        frame_detections = extract_bounding_boxes(valid_points, valid_classes)
        
        # 4. Feed into AB3DMOT Kalman Tracker
        active_tracks = tracker.update(frame_detections, dt=dt)
        
        print(f"--- Frame {i} ---")
        print(f"Raw Clusters Found: {len(frame_detections)}")
        print(f"Actively Tracked Objects: {len(active_tracks)}")
        
        # Print out tracked vehicles moving faster than 1 m/s
        for t in active_tracks:
            speed = np.sqrt(t['vx']**2 + t['vy']**2)
            if speed > 1.0:
                print(f"  -> [ID: {t['track_id']}] Class: {t['class_id']} | Speed: {speed:.1f} m/s | Pos: ({t['x']:.1f}, {t['y']:.1f})")

if __name__ == "__main__":
    run_tracking_pipeline()

import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

try:
    from trt_segformer import SegFormerTRT
except ImportError:
    print("❌ Error: trt_segformer.py not found.")

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini"
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
ENGINE_PATH = "./segformer_b1_python_build.engine"
OUTPUT_DIR = "phase1_geometry_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
PANO_W, PANO_H = 1024, 1024
CAM_DISPLAY_SIZE = 1024

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [90, 0, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

DYNAMIC_CLASSES = {
    11: {"name": "person", "min_pts": 5,  "min_l": 0.4, "max_l": 1.2, "min_w": 0.4, "max_w": 1.2, "max_h": 2.2},
    13: {"name": "car",    "min_pts": 15, "min_l": 2.0, "max_l": 6.0, "min_w": 1.5, "max_w": 2.8, "max_h": 2.5},
    14: {"name": "truck",  "min_pts": 30, "min_l": 4.0, "max_l": 12.0,"min_w": 2.0, "max_w": 3.5, "max_h": 4.5},
    15: {"name": "bus",    "min_pts": 40, "min_l": 5.0, "max_l": 15.0,"min_w": 2.5, "max_w": 3.5, "max_h": 4.5}
}
# ==========================================

class MinimalDataLoader:
    def __init__(self, json_dir, data_dir):
        self.data_dir = data_dir
        def load_json(name):
            with open(os.path.join(json_dir, name)) as f:
                return {i['token']: i for i in json.load(f)}
        self.samples = load_json("sample.json")
        self.sample_data = load_json("sample_data.json")
        self.calibrated_sensors = load_json("calibrated_sensor.json")
        self.ego_poses = load_json("ego_pose.json")
        with open(os.path.join(json_dir, "sensor.json")) as f:
            self.sensor_map = {s['token']: s['channel'] for s in json.load(f)}
        self.lookup = {}
        for sd in self.sample_data.values():
            s_token = sd['sample_token']
            calib = self.calibrated_sensors[sd['calibrated_sensor_token']]
            channel = self.sensor_map[calib['sensor_token']]
            if s_token not in self.lookup: self.lookup[s_token] = {}
            self.lookup[s_token][channel] = sd

    def get_data(self, sample_token, channel):
        sd = self.lookup[sample_token][channel]
        return {
            "path": os.path.join(self.data_dir, sd['filename']),
            "cs": self.calibrated_sensors[sd['calibrated_sensor_token']],
            "pose": self.ego_poses[sd['ego_pose_token']]
        }

def remove_ground(points, method="z"):
    """
    Step 1: Ground Removal.
    NuScenes LiDAR is mounted at ~1.8m height. 
    Z=0 is the LiDAR sensor. Ground is at approx -1.8m.
    """
    if method == "z":
        # Keep points above the road surface (-1.4m) and below tree-tops (2.5m)
        non_ground_mask = (points[:, 2] > -1.4) & (points[:, 2] < 2.5)
        return non_ground_mask
    elif method == "ransac":
        # Placeholder for future Phase 3 upgrade
        pass
    return np.ones(len(points), dtype=bool)

def fit_pca_obb(cluster_points, class_rules):
    """
    Step 4: Fit Oriented Bounding Box using PCA.
    Returns: [cx, cy, cz, l, w, h, yaw]
    """
    # 1. Center
    cx, cy, cz = np.mean(cluster_points, axis=0)
    
    # 2. XY Plane PCA for Orientation
    pca = PCA(n_components=2)
    pca.fit(cluster_points[:, :2])
    v1 = pca.components_[0]
    yaw = np.arctan2(v1[1], v1[0]) # Principal heading
    
    # 3. Rotate points to local bounding box frame
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    local_xy = np.dot(cluster_points[:, :2] - [cx, cy], rot.T)
    
    # 4. Extract physical dimensions
    l = np.max(local_xy[:, 0]) - np.min(local_xy[:, 0])
    w = np.max(local_xy[:, 1]) - np.min(local_xy[:, 1])
    h = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
    
    # 5. Semantic Geometric Verification
    if l > class_rules["max_l"] or w > class_rules["max_w"] or h > class_rules["max_h"]:
        return None # Blob is too big (segmentation smear or joined clusters)
        
    # Snap to minimums if LiDAR only hit one side of the car
    l = max(l, class_rules["min_l"])
    w = max(w, class_rules["min_w"])
    
    return [cx, cy, cz, l, w, h, yaw]

def get_8_corners_lidar_frame(bbox):
    cx, cy, cz, l, w, h, yaw = bbox
    dx, dy, dz = l/2, w/2, h/2
    x_corners = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
    y_corners = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    c, s = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    return np.dot(rot_mat, corners).T + np.array([cx, cy, cz])

def project_points(points, lidar_data, cam_data):
    R_l = Quaternion(lidar_data['cs']['rotation']).rotation_matrix
    T_l = np.array(lidar_data['cs']['translation'])
    pts = np.dot(R_l, points.T).T + T_l
    R_lp = Quaternion(lidar_data['pose']['rotation']).rotation_matrix
    T_lp = np.array(lidar_data['pose']['translation'])
    pts = np.dot(R_lp, pts.T).T + T_lp
    R_cp = Quaternion(cam_data['pose']['rotation']).rotation_matrix
    T_cp = np.array(cam_data['pose']['translation'])
    pts = np.dot(R_cp.T, (pts - T_cp).T).T
    R_c = Quaternion(cam_data['cs']['rotation']).rotation_matrix
    T_c = np.array(cam_data['cs']['translation'])
    pts = np.dot(R_c.T, (pts - T_c).T).T
    
    depths = pts[:, 2]
    mask = depths > 0.1
    K = np.array(cam_data['cs']['camera_intrinsic'])
    pts_img = np.dot(K, pts.T).T
    u = pts_img[:, 0] / np.where(depths == 0, 1e-6, depths)
    v = pts_img[:, 1] / np.where(depths == 0, 1e-6, depths)
    return np.vstack((u, v)).T, mask

def get_360_cylindrical_pixels(points_3d):
    thetas = np.arctan2(points_3d[:, 1], points_3d[:, 0])
    dist_xy = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    phis = np.arctan2(points_3d[:, 2], dist_xy)
    u = ((-thetas) / (2 * np.pi)) * PANO_W + (PANO_W / 2)
    u = np.mod(u, PANO_W)
    v = (PANO_H / 2) - (phis / np.radians(90)) * PANO_H
    return np.vstack((u, v)).T

def draw_3d_box_edges(img, corners_2d, color=(0, 255, 0)):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        p1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
        p2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))
        cv2.line(img, p1, p2, color, 2)

def process_phase1_frame(token, loader, trt_model):
    lidar_meta = loader.get_data(token, "LIDAR_TOP")
    raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
    num_pts = raw_lidar.shape[0]

    # Global arrays for tracking point semantics
    point_semantic_votes = np.full(num_pts, -1, dtype=np.int32)
    
    cam_data_cache = {}

    # --- PASS 1: Paint Points with Semantics ---
    for cam_name in CAMERAS:
        cam_meta = loader.get_data(token, cam_name)
        img = cv2.imread(cam_meta['path'])
        h, w = img.shape[:2]
        
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        pixels, depth_mask = project_points(raw_lidar, lidar_meta, cam_meta)
        valid_idx = np.where(depth_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                                          (pixels[:, 1] >= 0) & (pixels[:, 1] < h))[0]
        
        u_img, v_img = pixels[valid_idx, 0].astype(int), pixels[valid_idx, 1].astype(int)
        
        # Assign the semantic class to the LiDAR point
        point_semantic_votes[valid_idx] = seg_mask[v_img, u_img]
        
        cam_data_cache[cam_name] = {
            "meta": cam_meta, "img": img, "seg_mask": seg_mask, "pixels": pixels
        }

    # --- PASS 2: Ground Removal ---
    non_ground_mask = remove_ground(raw_lidar, method="z")
    valid_indices = np.where(non_ground_mask)[0]
    
    geom_pts = raw_lidar[valid_indices]
    geom_votes = point_semantic_votes[valid_indices]

    # --- PASS 3: Geometry-First Clustering ---
    detected_objects = []
    if len(geom_pts) > 0:
        # Cluster ALL non-ground points regardless of class
        clustering = DBSCAN(eps=0.8, min_samples=5).fit(geom_pts)
        
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            if label == -1: continue # Drop unclustered noise
            
            cluster_mask = (clustering.labels_ == label)
            cluster_pts = geom_pts[cluster_mask]
            cluster_votes = geom_votes[cluster_mask]
            
            # --- PASS 4: Semantic Confirmation (Voting) ---
            # Filter out points that weren't painted (value is -1)
            valid_votes = cluster_votes[cluster_votes != -1]
            if len(valid_votes) == 0: continue # Purely structural background (no semantic hit)
            
            # Majority Vote
            counts = np.bincount(valid_votes)
            winning_class = np.argmax(counts)
            
            # Is the winning class an object we care about?
            if winning_class in DYNAMIC_CLASSES:
                rules = DYNAMIC_CLASSES[winning_class]
                
                # Minimum Point Threshold for this specific class
                if len(cluster_pts) >= rules["min_pts"]:
                    
                    # --- PASS 5: PCA OBB Fitting ---
                    bbox = fit_pca_obb(cluster_pts, rules)
                    if bbox is not None:
                        detected_objects.append({
                            "class": rules["name"],
                            "bbox": bbox
                        })

    # --- RENDER PASS ---
    # We will draw the 360-Pano and just ONE front camera to keep the image clean
    pano_pane = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
    pano_coords = get_360_cylindrical_pixels(raw_lidar)
    point_depths = np.linalg.norm(raw_lidar, axis=1)
    
    # Draw Background LiDAR
    for i in range(num_pts):
        u, v = int(pano_coords[i, 0]), int(pano_coords[i, 1])
        if 0 <= u < PANO_W and 0 <= v < PANO_H and point_depths[i] < 60:
            val = int(np.clip(60 - point_depths[i], 10, 60))
            pano_pane[v, u] = [val, val, val]
            
    # Draw Validated OBBs on Pano
    for obj in detected_objects:
        corners_3d = get_8_corners_lidar_frame(obj["bbox"])
        corners_pano = get_360_cylindrical_pixels(corners_3d)
        draw_3d_box_edges(pano_pane, corners_pano, color=(0, 255, 0))

    cv2.putText(pano_pane, "360 LIDAR | ROBUST PCA GEOMETRY", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Front Cam Render
    cam_name = "CAM_FRONT"
    cache = cam_data_cache[cam_name]
    driver_img = cv2.imread(cache['meta']['path'])
    
    color_mask = CITYSCAPES_COLORS[cache['seg_mask']]
    blended = cv2.addWeighted(driver_img, 0.7, color_mask, 0.3, 0)
    
    # Project 3D Boxes to Camera
    for obj in detected_objects:
        corners_3d = get_8_corners_lidar_frame(obj["bbox"])
        corners_cam, depth_mask = project_points(corners_3d, lidar_meta, cache['meta'])
        
        if np.sum(depth_mask) >= 4:
            draw_3d_box_edges(blended, corners_cam, color=(0, 255, 0))
            if depth_mask[0]:
                cv2.putText(blended, obj["class"].upper(), 
                            (int(corners_cam[0,0]), int(corners_cam[0,1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    blended = cv2.resize(blended, (CAM_DISPLAY_SIZE, CAM_DISPLAY_SIZE))
    cv2.putText(blended, f"CAM_FRONT | Detections: {len(detected_objects)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return cv2.hconcat([pano_pane, blended])

if __name__ == "__main__":
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    tokens = list(loader.samples.keys())[:15]

    for i, token in enumerate(tokens):
        print(f"[{i+1}/{len(tokens)}] Processing Robust Geometry for Token: {token[:8]}...")
        panorama = process_phase1_frame(token, loader, trt_model)
        
        out_path = os.path.join(OUTPUT_DIR, f"phase1_geom_{i:03d}.jpg")
        cv2.imwrite(out_path, panorama)
    
    print(f"\n✅ Phase 1 single-frame geometry saved to {OUTPUT_DIR}")

import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Import your TensorRT Segformer wrapper
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
OUTPUT_VIDEO = "clustering_3d_fusion.mp4"

# Dimensions
PANO_W, PANO_H = 1024, 1024
CAM_DISPLAY_SIZE = 1024
FPS = 2  

PAINT_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [90, 0, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

# Clustering Priors
CLASS_PRIORS = {
    "car":        {"l": 4.5, "w": 1.9, "h": 1.5, "min_pts": 15},
    "truck":      {"l": 8.0, "w": 2.5, "h": 3.0, "min_pts": 25},
    "bus":        {"l": 10.0, "w": 2.8, "h": 3.5, "min_pts": 30},
    "person":     {"l": 0.6, "w": 0.6, "h": 1.7, "min_pts": 5},
    "motorcycle": {"l": 2.0, "w": 0.8, "h": 1.5, "min_pts": 8},
    "bicycle":    {"l": 1.8, "w": 0.6, "h": 1.5, "min_pts": 8}
}

DYNAMIC_CLASSES = {
    11: "person", 13: "car", 14: "truck", 
    15: "bus", 17: "motorcycle", 18: "bicycle"
}
# ==========================================

class MinimalDataLoader:
    # (Remains unchanged)
    def __init__(self, json_dir, data_dir):
        self.data_dir = data_dir
        def load_json(name):
            with open(os.path.join(json_dir, name)) as f:
                return {item['token']: item for item in json.load(f)}
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
        if sample_token not in self.lookup or channel not in self.lookup[sample_token]:
            return None
        sd = self.lookup[sample_token][channel]
        return {
            "path": os.path.join(self.data_dir, sd['filename']),
            "cs": self.calibrated_sensors[sd['calibrated_sensor_token']],
            "pose": self.ego_poses[sd['ego_pose_token']]
        }

def project_points(points, lidar_data, cam_data):
    # Transformation matrices
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

def fit_3d_bbox(cluster_points, class_name):
    """Fits an OBB and applies Class Priors for volume approximation."""
    center = np.mean(cluster_points, axis=0)
    priors = CLASS_PRIORS[class_name]
    
    pca = PCA(n_components=2)
    pca.fit(cluster_points[:, :2])
    heading_vec = pca.components_[0]
    yaw = np.arctan2(heading_vec[1], heading_vec[0])
    
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    local_pts = np.dot(cluster_points[:, :2] - center[:2], rot_mat.T)
    
    raw_l = np.max(local_pts[:, 0]) - np.min(local_pts[:, 0])
    raw_w = np.max(local_pts[:, 1]) - np.min(local_pts[:, 1])
    raw_h = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
    
    final_l = priors["l"] if raw_l < (priors["l"] * 0.6) else (raw_l + 0.2)
    final_w = priors["w"] if raw_w < (priors["w"] * 0.6) else (raw_w + 0.2)
    final_h = priors["h"] if raw_h < (priors["h"] * 0.6) else raw_h
    
    return [center[0], center[1], center[2], final_l, final_w, final_h, yaw]

def get_3d_box_corners(bbox):
    """Converts [cx, cy, cz, l, w, h, yaw] into 8 corner points in 3D."""
    cx, cy, cz, l, w, h, yaw = bbox
    dx, dy, dz = l/2, w/2, h/2
    
    # 8 corners in local bounding box frame
    x_corners = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
    y_corners = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotate by yaw around Z axis
    c, s = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    rotated_corners = np.dot(rot_mat, corners).T
    # Translate to center
    global_corners = rotated_corners + np.array([cx, cy, cz])
    return global_corners

def draw_3d_box_edges(img, corners_2d, color=(0, 255, 0), thickness=2, is_cylindrical=False):
    """Draws the 12 edges of a 3D box given its 8 projected 2D corners."""
    # Box edges definitions (indices of the 8 corners)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Top face
        (4, 5), (5, 6), (6, 7), (7, 4), # Bottom face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
    ]
    
    for i, j in edges:
        p1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
        p2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))
        
        # Guard against wrapping lines in the 360 projection
        if is_cylindrical and abs(p1[0] - p2[0]) > PANO_W / 2:
            continue
            
        cv2.line(img, p1, p2, color, thickness)

def run_visual_analysis():
    print("Loading Core Systems...")
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    
    sample_tokens = list(loader.samples.keys())[:50] 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (PANO_W + CAM_DISPLAY_SIZE, CAM_DISPLAY_SIZE))

    for idx, token in enumerate(sample_tokens):
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        cam_front_meta = loader.get_data(token, "CAM_FRONT")
        if not lidar_meta or not cam_front_meta: continue
            
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        num_pts = raw_lidar.shape[0]
        
        point_colors = np.zeros((num_pts, 3), dtype=np.uint8)
        point_assigned = np.zeros(num_pts, dtype=bool)
        point_class_ids = np.full(num_pts, -1, dtype=np.int32) # Store class ID for clustering
        point_depths = np.linalg.norm(raw_lidar, axis=1)

        # 1. Semantic Painting (FRONT ONLY)
        for cam_name in PAINT_CAMERAS:
            cam_meta = loader.get_data(token, cam_name)
            if not cam_meta: continue
            
            img = cv2.imread(cam_meta['path'])
            h, w = img.shape[:2]
            logits, _ = trt_model.infer(img)
            seg_mask = np.argmax(logits, axis=1)[0]
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            pixels, depth_mask = project_points(raw_lidar, lidar_meta, cam_meta)
            valid_fov = (depth_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                                     (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
            
            v_idx = np.where(valid_fov)[0]
            if len(v_idx) == 0: continue
            
            u_img, v_img = pixels[v_idx, 0].astype(int), pixels[v_idx, 1].astype(int)
            classes = seg_mask[v_img, u_img]
            
            point_class_ids[v_idx] = classes # Save class ID for DBSCAN
            point_colors[v_idx] = CITYSCAPES_COLORS[classes]
            point_assigned[v_idx] = True

        # 2. Density Clustering on Painted Points
        detected_objects = []
        for class_idx, class_name in DYNAMIC_CLASSES.items():
            mask = (point_class_ids == class_idx)
            target_pts = raw_lidar[mask]
            
            min_pts = CLASS_PRIORS[class_name]["min_pts"]
            if len(target_pts) < min_pts: continue
                
            clustering = DBSCAN(eps=0.8, min_samples=min_pts).fit(target_pts)
            for label in set(clustering.labels_):
                if label == -1: continue
                cluster_pts = target_pts[clustering.labels_ == label]
                if len(cluster_pts) < min_pts: continue
                    
                bbox = fit_3d_bbox(cluster_pts, class_name)
                detected_objects.append({"class": class_name, "bbox": bbox})

        # 3. Render Left Pane (Lidar 360)
        pano_pane = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
        pano_coords = get_360_cylindrical_pixels(raw_lidar)
        
        for i in range(num_pts):
            u, v = int(pano_coords[i, 0]), int(pano_coords[i, 1])
            if 0 <= u < PANO_W and 0 <= v < PANO_H:
                dist = point_depths[i]
                if dist > 80: continue
                
                if point_assigned[i]:
                    intensity = np.clip(1.3 - (dist / 50.0), 0.4, 1.0)
                    color = (point_colors[i] * intensity).astype(np.uint8)
                    cv2.circle(pano_pane, (u, v), 2, color.tolist(), -1)
                else:
                    val = int(np.clip(60 - dist, 10, 60))
                    pano_pane[v, u] = [val, val, val]
                    
        # Draw 3D Bounding Boxes on Pano View
        for obj in detected_objects:
            corners_3d = get_3d_box_corners(obj["bbox"])
            corners_pano = get_360_cylindrical_pixels(corners_3d)
            draw_3d_box_edges(pano_pane, corners_pano, color=(0, 255, 0), thickness=2, is_cylindrical=True)

        sectors = ["BACK", "BACK LEFT", "FRONT LEFT", "FRONT", "FRONT RIGHT", "BACK RIGHT", "BACK"]
        for i, name in enumerate(sectors):
            x = int((i / 6.0) * PANO_W)
            cv2.line(pano_pane, (x, 0), (x, PANO_H), (100, 100, 100), 1)
            if i < 6:
                cv2.putText(pano_pane, name, (x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Render Right Pane (Camera Image with 3D Boxes)
        driver_img = cv2.imread(cam_front_meta['path'])
        
        # Project 3D Bounding Boxes onto Camera Image
        for obj in detected_objects:
            corners_3d = get_3d_box_corners(obj["bbox"])
            corners_cam, depth_mask = project_points(corners_3d, lidar_meta, cam_front_meta)
            
            # Only draw if the entire box is mostly in front of the camera
            if np.sum(depth_mask) >= 4: 
                draw_3d_box_edges(driver_img, corners_cam, color=(0, 255, 0), thickness=3)
                
                # Draw class label on top-left corner of the box
                top_left_corner = corners_cam[0]
                if depth_mask[0]:
                    cv2.putText(driver_img, obj["class"].upper(), 
                                (int(top_left_corner[0]), int(top_left_corner[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        driver_img = cv2.resize(driver_img, (CAM_DISPLAY_SIZE, CAM_DISPLAY_SIZE))
        
        # 5. Combine and Finish
        final_frame = np.hstack((pano_pane, driver_img))
        cv2.putText(final_frame, f"Analysis: {token[:8]} | FRONT-PAINTED ONLY | {len(detected_objects)} Objects", 
                    (50, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out_video.write(final_frame)
        if (idx+1) % 5 == 0: print(f"Encoded {idx+1} frames...")

    out_video.release()
    print(f"\n✅ Analysis video saved: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_visual_analysis()

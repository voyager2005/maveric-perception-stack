import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN

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
OUTPUT_DIR = "global_map_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

# Map Canvas Config
MAP_SIZE = 1000
PIXELS_PER_METER = 10  # 10 pixels = 1 meter. Map covers 100x100 meters.

DYNAMIC_CLASSES = {
    11: {"name": "person", "l_prior": 0.6, "min_w": 0.2, "max_w": 1.2, "min_h": 0.5, "max_h": 2.2},
    13: {"name": "car",    "l_prior": 4.5, "min_w": 1.0, "max_w": 2.8, "min_h": 0.5, "max_h": 2.8},
    14: {"name": "truck",  "l_prior": 8.0, "min_w": 1.5, "max_w": 3.5, "min_h": 1.0, "max_h": 4.5},
    15: {"name": "bus",    "l_prior": 10.0,"min_w": 1.5, "max_w": 3.5, "min_h": 1.0, "max_h": 4.5}
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

def remove_ground(points):
    non_ground_mask = points[:, 2] > -1.4
    return points[non_ground_mask]

def project_lidar_to_cam(points, lidar_data, cam_data):
    """Lidar to Cam Sensor Pixels"""
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
    return np.vstack((u, v)).T, mask, depths

def build_3d_box_from_2d(u, v, w_2d, h_2d, median_depth, num_pts, K, class_rules):
    required_pts = max(1, int(20.0 / median_depth))
    if num_pts < required_pts: return None
    fx, fy = K[0, 0], K[1, 1]
    cx_img, cy_img = K[0, 2], K[1, 2]
    
    Z = median_depth
    X = (u - cx_img) * Z / fx
    Y = (v - cy_img) * Z / fy
    
    W_3d = (w_2d * Z) / fx
    H_3d = (h_2d * Z) / fy
    L_3d = class_rules["l_prior"] 
    
    if W_3d < class_rules["min_w"] or W_3d > class_rules["max_w"]: return None
    if H_3d < class_rules["min_h"] or H_3d > class_rules["max_h"]: return None
        
    return [X, Y, Z]

def transform_cam_to_global(cam_pts_3d, cam_meta):
    """
    CRITICAL MATH: Pushes a 3D point from the Camera Lens up to the Global Map.
    """
    pts = np.array(cam_pts_3d).reshape(-1, 3)
    
    # 1. Camera Frame -> Ego Vehicle Frame
    R_cs = Quaternion(cam_meta['cs']['rotation']).rotation_matrix
    T_cs = np.array(cam_meta['cs']['translation'])
    ego_pts = np.dot(R_cs, pts.T).T + T_cs
    
    # 2. Ego Vehicle Frame -> Global Map Frame (Using IMU / ego_pose)
    R_pose = Quaternion(cam_meta['pose']['rotation']).rotation_matrix
    T_pose = np.array(cam_meta['pose']['translation'])
    global_pts = np.dot(R_pose, ego_pts.T).T + T_pose
    
    return global_pts[0] # Return the single [X, Y, Z] point

def run_global_map_test():
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    tokens = list(loader.samples.keys())[:30] # Let's run 30 frames to see movement
    
    # History Arrays
    ego_trajectory_global = []
    object_detections_global = []

    # Get the Global position of the car in Frame 0. 
    # We will use this as the center of our Canvas (0,0) so the coordinates fit on screen.
    first_ego_pose = loader.get_data(tokens[0], "CAM_FRONT")['pose']['translation']
    map_origin_x, map_origin_y = first_ego_pose[0], first_ego_pose[1]

    for i, token in enumerate(tokens):
        print(f"[{i+1}/{len(tokens)}] Extracting Global Coordinates for Token: {token[:8]}...")
        
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        filtered_lidar = remove_ground(raw_lidar)

        # Log Ego Vehicle Global Position (using CAM_FRONT timestamp)
        ego_pos = loader.get_data(token, "CAM_FRONT")['pose']['translation']
        ego_trajectory_global.append((ego_pos[0], ego_pos[1]))

        for cam_name in CAMERAS:
            cam_meta = loader.get_data(token, cam_name)
            img = cv2.imread(cam_meta['path'])
            h, w = img.shape[:2]
            
            logits, _ = trt_model.infer(img)
            seg_mask = np.argmax(logits, axis=1)[0]
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            pixels, depth_mask, depths = project_lidar_to_cam(filtered_lidar, lidar_meta, cam_meta)
            K = np.array(cam_meta['cs']['camera_intrinsic'])
            
            for class_id, rules in DYNAMIC_CLASSES.items():
                binary_mask = np.uint8(seg_mask == class_id)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                for lbl_idx in range(1, num_labels): 
                    u_min, v_min, w_2d, h_2d, area = stats[lbl_idx]
                    u_center, v_center = centroids[lbl_idx]
                    
                    if area < 100: continue
                    
                    in_box = (depth_mask) & (pixels[:, 0] >= u_min) & (pixels[:, 0] <= u_min + w_2d) & \
                                            (pixels[:, 1] >= v_min) & (pixels[:, 1] <= v_min + h_2d)
                    
                    pts_inside = np.where(in_box)[0]
                    
                    if len(pts_inside) > 0:
                        depths_in_box = depths[pts_inside]
                        if len(depths_in_box) > 2:
                            d_clust = DBSCAN(eps=1.5, min_samples=1).fit(depths_in_box.reshape(-1, 1))
                            unique_labels, counts = np.unique(d_clust.labels_, return_counts=True)
                            best_label = unique_labels[np.argmax(counts)]
                            median_depth = np.median(depths_in_box[d_clust.labels_ == best_label])
                        else:
                            median_depth = np.median(depths_in_box)
                            
                        # 1. Get object center in Local Camera Frame
                        center_cam = build_3d_box_from_2d(u_center, v_center, w_2d, h_2d, median_depth, len(pts_inside), K, rules)
                        
                        if center_cam is not None:
                            # 2. THE MAGIC: Push the Local object into the Global Map Frame
                            center_global = transform_cam_to_global(center_cam, cam_meta)
                            
                            object_detections_global.append({
                                "class": rules["name"],
                                "global_x": center_global[0],
                                "global_y": center_global[1],
                                "frame_idx": i
                            })

        # --- RENDER THE AGGREGATE MAP FOR THIS FRAME ---
        map_canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        cx, cy = MAP_SIZE // 2, MAP_SIZE // 2 # Canvas Center
        
        # Helper to convert Global (X,Y) to Canvas Pixel (U,V)
        def to_pixel(gx, gy):
            # We subtract the very first frame's origin so the map stays centered
            px = int(cx + (gx - map_origin_x) * PIXELS_PER_METER)
            py = int(cy - (gy - map_origin_y) * PIXELS_PER_METER) # -Y because images draw top-down
            return px, py

        # Draw Grid Lines
        for step in range(0, MAP_SIZE, PIXELS_PER_METER * 10): # 10 meter grid lines
            cv2.line(map_canvas, (0, step), (MAP_SIZE, step), (30, 30, 30), 1)
            cv2.line(map_canvas, (step, 0), (step, MAP_SIZE), (30, 30, 30), 1)

        # Draw Ego Trajectory History (Blue Line)
        for t in range(1, len(ego_trajectory_global)):
            pt1 = to_pixel(*ego_trajectory_global[t-1])
            pt2 = to_pixel(*ego_trajectory_global[t])
            cv2.line(map_canvas, pt1, pt2, (255, 0, 0), 2)
            
        # Draw current Ego Car (Big Blue Dot)
        curr_ego_px = to_pixel(*ego_trajectory_global[-1])
        cv2.circle(map_canvas, curr_ego_px, 5, (255, 100, 100), -1)

        # Draw Object Detection History (Red Dots)
        for obj in object_detections_global:
            px, py = to_pixel(obj["global_x"], obj["global_y"])
            
            # Fade older detections to see movement/stability
            age = i - obj["frame_idx"]
            intensity = max(50, 255 - (age * 15))
            
            color = (0, 0, intensity) if obj["class"] == "car" else (0, intensity, intensity)
            cv2.circle(map_canvas, (px, py), 3, color, -1)

        cv2.putText(map_canvas, f"Global Map Verification | Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(map_canvas, f"Grid = 10m | Ego = Blue | Cars = Red", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        out_path = os.path.join(OUTPUT_DIR, f"global_map_{i:03d}.jpg")
        cv2.imwrite(out_path, map_canvas)

    print(f"\n✅ Global Transform maps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_global_map_test()

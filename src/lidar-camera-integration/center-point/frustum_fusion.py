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
OUTPUT_DIR = "frustum_fusion_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [90, 0, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

DYNAMIC_CLASSES = {
    11: {"name": "person", "l_prior": 0.6, "max_w": 1.2, "max_h": 2.2},
    13: {"name": "car",    "l_prior": 4.5, "max_w": 2.5, "max_h": 2.5},
    14: {"name": "truck",  "l_prior": 8.0, "max_w": 3.5, "max_h": 4.5},
    15: {"name": "bus",    "l_prior": 10.0,"max_w": 3.5, "max_h": 4.5}
}
# ==========================================

class ObjectTracker:
    """Lightweight 3D tracking memory to retain objects across frames."""
    def __init__(self, max_age=3, distance_threshold=3.0):
        self.tracks = [] 
        self.max_age = max_age
        self.dist_thresh = distance_threshold
        self.next_id = 0

    def update(self, new_objects):
        # Age all tracks
        for t in self.tracks: t['age'] += 1
        
        for obj in new_objects:
            cx, cy, cz = obj['center_3d']
            matched = False
            for t in self.tracks:
                t_cx, t_cy, t_cz = t['center_3d']
                dist = np.sqrt((cx-t_cx)**2 + (cy-t_cy)**2 + (cz-t_cz)**2)
                
                if dist < self.dist_thresh and t['class'] == obj['class']:
                    # Update track
                    t['center_3d'] = obj['center_3d']
                    t['dims'] = obj['dims']
                    t['age'] = 0
                    matched = True
                    break
            
            if not matched:
                obj['id'] = self.next_id
                obj['age'] = 0
                self.tracks.append(obj)
                self.next_id += 1
                
        # Keep tracks that are young enough
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        # Return currently visible objects (age == 0) and slightly retained ones
        return [t for t in self.tracks if t['age'] <= 1]

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

def project_points(points, lidar_data, cam_data):
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

def build_3d_box_from_2d(u, v, w_2d, h_2d, median_depth, K, class_rules):
    """Unprojects a 2D bounding box back into 3D Camera Space using Depth."""
    fx, fy = K[0, 0], K[1, 1]
    cx_img, cy_img = K[0, 2], K[1, 2]
    
    Z = median_depth
    X = (u - cx_img) * Z / fx
    Y = (v - cy_img) * Z / fy
    
    W_3d = (w_2d * Z) / fx
    H_3d = (h_2d * Z) / fy
    L_3d = class_rules["l_prior"] 
    
    if W_3d > class_rules["max_w"] or H_3d > class_rules["max_h"]:
        return None
        
    return {"center_cam": [X, Y, Z], "dims": [W_3d, H_3d, L_3d]}

def get_8_corners_cam_frame(box_3d):
    """Generates 8 corners of the box in the Camera coordinate frame."""
    # 🐛 FIX: Changed "center_cam" to "center_3d" to match the tracker format
    X, Y, Z = box_3d["center_3d"] 
    W, H, L = box_3d["dims"]
    
    dx, dy, dz = W/2, H/2, L/2
    x_corners = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
    y_corners = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
    
    corners = np.vstack([x_corners, y_corners, z_corners]).T
    return corners + np.array([X, Y, Z])

def draw_3d_box_edges(img, corners_2d, color=(0, 255, 0)):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        p1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
        p2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))
        cv2.line(img, p1, p2, color, 2)

def process_frustum_frame(token, loader, trt_model, trackers):
    cam_results = []
    lidar_meta = loader.get_data(token, "LIDAR_TOP")
    raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]

    for cam_name in CAMERAS:
        cam_meta = loader.get_data(token, cam_name)
        img = cv2.imread(cam_meta['path'])
        h, w = img.shape[:2]
        
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        color_mask = CITYSCAPES_COLORS[seg_mask]
        blended = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        
        pixels, depth_mask, depths = project_points(raw_lidar, lidar_meta, cam_meta)
        K = np.array(cam_meta['cs']['camera_intrinsic'])
        
        frame_objects = []
        
        for class_id, rules in DYNAMIC_CLASSES.items():
            binary_mask = np.uint8(seg_mask == class_id)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            for i in range(1, num_labels): 
                u_min, v_min, w_2d, h_2d, area = stats[i]
                u_center, v_center = centroids[i]
                
                if area < 100: continue
                
                in_box = (depth_mask) & (pixels[:, 0] >= u_min) & (pixels[:, 0] <= u_min + w_2d) & \
                                        (pixels[:, 1] >= v_min) & (pixels[:, 1] <= v_min + h_2d)
                
                pts_inside = np.where(in_box)[0]
                
                if len(pts_inside) > 0:
                    depths_in_box = depths[pts_inside]
                    
                    if len(depths_in_box) > 2:
                        d_clust = DBSCAN(eps=2.0, min_samples=1).fit(depths_in_box.reshape(-1, 1))
                        unique_labels, counts = np.unique(d_clust.labels_, return_counts=True)
                        best_label = unique_labels[np.argmax(counts)]
                        true_depths = depths_in_box[d_clust.labels_ == best_label]
                        median_depth = np.median(true_depths)
                    else:
                        median_depth = np.median(depths_in_box)
                        
                    box_3d = build_3d_box_from_2d(u_center, v_center, w_2d, h_2d, median_depth, K, rules)
                    
                    if box_3d is not None:
                        frame_objects.append({
                            "class": rules["name"],
                            "center_3d": box_3d["center_cam"], 
                            "dims": box_3d["dims"]
                        })

        active_objects = trackers[cam_name].update(frame_objects)

        for obj in active_objects:
            corners_cam = get_8_corners_cam_frame(obj)
            
            # Use intrinsic matrix to draw lines on image
            u_c = (corners_cam[:, 0] * K[0, 0] / corners_cam[:, 2]) + K[0, 2]
            v_c = (corners_cam[:, 1] * K[1, 1] / corners_cam[:, 2]) + K[1, 2]
            corners_2d = np.vstack((u_c, v_c)).T
            
            draw_3d_box_edges(blended, corners_2d, color=(0, 255, 0))
            
            cv2.putText(blended, f"{obj['class'].upper()} ID:{obj.get('id', 'N')} ({obj['center_3d'][2]:.1f}m)", 
                        (int(corners_2d[0,0]), int(corners_2d[0,1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(blended, f"{cam_name} | Tracks: {len(active_objects)}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cam_results.append(blended)

    return cv2.hconcat(cam_results)

if __name__ == "__main__":
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    tokens = list(loader.samples.keys())[:15]
    
    trackers = {cam: ObjectTracker(max_age=3, distance_threshold=2.5) for cam in CAMERAS}

    for i, token in enumerate(tokens):
        print(f"[{i+1}/{len(tokens)}] Frustum Fusion on Token: {token[:8]}...")
        panorama = process_frustum_frame(token, loader, trt_model, trackers)
        
        out_path = os.path.join(OUTPUT_DIR, f"frustum_{i:03d}.jpg")
        cv2.imwrite(out_path, panorama)
    
    print(f"\n✅ Frustum visualizations saved to {OUTPUT_DIR}")

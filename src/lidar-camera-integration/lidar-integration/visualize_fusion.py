import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion

# Import your TensorRT Segformer wrapper
# (Ensure trt_segformer.py is in the same folder, or adjust your path)
from trt_segformer import SegFormerTRT

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
# Adjust these paths to your old directory structure
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini" 
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
ENGINE_PATH = "./segformer_b1_python_build.engine"
OUTPUT_DIR = "fusion_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard Cityscapes 19-Class Color Map (BGR format for OpenCV)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],  # 0: road
    [232, 35, 244],  # 1: sidewalk
    [70, 70, 70],    # 2: building
    [156, 102, 102], # 3: wall
    [153, 153, 190], # 4: fence
    [153, 153, 153], # 5: pole
    [30, 170, 250],  # 6: traffic light
    [0, 220, 220],   # 7: traffic sign
    [35, 142, 107],  # 8: vegetation
    [152, 251, 152], # 9: terrain
    [180, 130, 70],  # 10: sky
    [60, 20, 220],   # 11: person
    [0, 0, 255],     # 12: rider
    [142, 0, 0],     # 13: car
    [70, 0, 0],      # 14: truck
    [100, 60, 0],    # 15: bus
    [90, 0, 0],      # 16: train
    [230, 0, 0],     # 17: motorcycle
    [32, 11, 119]    # 18: bicycle
], dtype=np.uint8)

# Cityscapes Class Names for the Legend
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", 
    "truck", "bus", "train", "motorcycle", "bicycle"
]
# ==========================================

class MinimalDataLoader:
    """Self-contained JSON parser for visualization"""
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

        # Precompute lookups
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
    """Rigid body transformation from LiDAR to Camera pixels"""
    # 1. Lidar to Ego
    R_l = Quaternion(lidar_data['cs']['rotation']).rotation_matrix
    T_l = np.array(lidar_data['cs']['translation'])
    pts = np.dot(R_l, points.T).T + T_l
    
    # 2. Ego to Global
    R_lp = Quaternion(lidar_data['pose']['rotation']).rotation_matrix
    T_lp = np.array(lidar_data['pose']['translation'])
    pts = np.dot(R_lp, pts.T).T + T_lp
    
    # 3. Global to Cam Ego
    R_cp = Quaternion(cam_data['pose']['rotation']).rotation_matrix
    T_cp = np.array(cam_data['pose']['translation'])
    pts = np.dot(R_cp.T, (pts - T_cp).T).T
    
    # 4. Cam Ego to Cam Frame
    R_c = Quaternion(cam_data['cs']['rotation']).rotation_matrix
    T_c = np.array(cam_data['cs']['translation'])
    pts = np.dot(R_c.T, (pts - T_c).T).T
    
    depths = pts[:, 2]
    mask = depths > 0.1
    
    # 5. Intrinsic Projection
    K = np.array(cam_data['cs']['camera_intrinsic'])
    pts_img = np.dot(K, pts.T).T
    
    u = pts_img[:, 0] / np.where(depths == 0, 1e-6, depths)
    v = pts_img[:, 1] / np.where(depths == 0, 1e-6, depths)
    
    return np.vstack((u, v)).T, mask

def run_visualization():
    print("Loading Data & TensorRT Engine...")
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    
    sample_tokens = list(loader.samples.keys())[:5] # Just grab the first 5 sweeps
    
    for i, token in enumerate(sample_tokens):
        print(f"Processing Sweep {i+1}/5...")
        
        # Fetch Lidar and Front Camera data
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        cam_meta = loader.get_data(token, "CAM_FRONT")
        
        # 1. Load and Run Segformer
        img = cv2.imread(cam_meta['path'])
        h, w = img.shape[:2]
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Create the Opaque Semantic Overlay
        color_mask = CITYSCAPES_COLORS[seg_mask]
        # Blend: 60% Original Image, 40% Semantic Mask
        blended_img = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
        
        # 3. Load and Project LiDAR
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        pixels, depth_mask = project_points(raw_lidar, lidar_meta, cam_meta)
        
        # Filter points inside the camera view
        valid_idx = np.where(depth_mask & 
                             (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                             (pixels[:, 1] >= 0) & (pixels[:, 1] < h))[0]
                             
        valid_u = pixels[valid_idx, 0].astype(int)
        valid_v = pixels[valid_idx, 1].astype(int)
        
        # 4. Draw the Color-Coded LiDAR Points
        for u, v in zip(valid_u, valid_v):
            # Look up the semantic class at this exact pixel
            class_id = seg_mask[v, u]
            point_color = CITYSCAPES_COLORS[class_id].tolist()
            
            # Draw the point with a tiny white border so it pops against the opaque background
            cv2.circle(blended_img, (u, v), 3, (255, 255, 255), -1) # Border
            cv2.circle(blended_img, (u, v), 2, point_color, -1)     # Semantic Color

        # 5. Add a mini-legend for dynamic classes (Cars, Pedestrians, Bicycles)
        legend_y = 30
        for class_idx in [11, 13, 18]: # Person, Car, Bicycle
            color = CITYSCAPES_COLORS[class_idx].tolist()
            cv2.rectangle(blended_img, (20, legend_y - 15), (40, legend_y + 5), color, -1)
            cv2.putText(blended_img, CLASS_NAMES[class_idx], (50, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            legend_y += 35

        out_path = os.path.join(OUTPUT_DIR, f"sweep_{i+1}_fusion.jpg")
        cv2.imwrite(out_path, blended_img)
        print(f"  -> Saved {out_path}")

if __name__ == "__main__":
    run_visualization()

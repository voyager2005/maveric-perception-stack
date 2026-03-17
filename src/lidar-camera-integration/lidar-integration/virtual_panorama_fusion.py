import os
import json
import numpy as np
import cv2
import glob
from pyquaternion import Quaternion

# Import your TensorRT Segformer wrapper
# Ensure trt_segformer.py is in the same directory
try:
    from trt_segformer import SegFormerTRT
except ImportError:
    print("❌ Error: trt_segformer.py not found in current directory.")

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini"
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
ENGINE_PATH = "./segformer_b1_python_build.engine"
OUTPUT_DIR = "virtual_panorama_results"

# Panorama Resolution
PANO_W = 1920
PANO_H = 720
FOV_H = 180  # Horizontal Field of View in degrees
FOV_V = 60   # Vertical Field of View in degrees

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Order cameras for a logical sweep from left to right
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

# Cityscapes Color Palette (BGR)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [90, 0, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

# ==========================================

class MinimalDataLoader:
    def __init__(self, json_dir, data_dir):
        self.data_dir = data_dir
        print(f"Loading metadata from {json_dir}...")
        
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
    """3D LiDAR -> 2D Camera Pixel Projection"""
    # 1. Lidar to Ego
    R_l = Quaternion(lidar_data['cs']['rotation']).rotation_matrix
    T_l = np.array(lidar_data['cs']['translation'])
    pts = np.dot(R_l, points.T).T + T_l
    
    # 2. Ego to Global
    R_lp = Quaternion(lidar_data['pose']['rotation']).rotation_matrix
    T_lp = np.array(lidar_data['pose']['translation'])
    pts = np.dot(R_lp, pts.T).T + T_lp
    
    # 3. Global to Cam Ego (At Camera Timestamp)
    R_cp = Quaternion(cam_data['pose']['rotation']).rotation_matrix
    T_cp = np.array(cam_data['pose']['translation'])
    pts = np.dot(R_cp.T, (pts - T_cp).T).T
    
    # 4. Cam Ego to Cam Sensor Frame
    R_c = Quaternion(cam_data['cs']['rotation']).rotation_matrix
    T_c = np.array(cam_data['cs']['translation'])
    pts = np.dot(R_c.T, (pts - T_c).T).T
    
    depths = pts[:, 2]
    mask = depths > 0.1 # Point must be in front
    
    # 5. Intrinsic Projection
    K = np.array(cam_data['cs']['camera_intrinsic'])
    pts_img = np.dot(K, pts.T).T
    
    u = pts_img[:, 0] / np.where(depths == 0, 1e-6, depths)
    v = pts_img[:, 1] / np.where(depths == 0, 1e-6, depths)
    
    return np.vstack((u, v)).T, mask, depths

def get_cylindrical_pixels(points_3d):
    """Projects 3D points onto a 2D cylindrical panorama strip with Center-Forward alignment"""
    # Horizontal Angle (Yaw)
    # np.arctan2(y, x) returns 0 for straight ahead (X-axis).
    # We want 0 (the center of our image) to be the X-axis.
    thetas = np.arctan2(points_3d[:, 1], points_3d[:, 0])
    
    # Vertical Angle (Pitch)
    dist_xy = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    phis = np.arctan2(points_3d[:, 2], dist_xy)
    
    # Map Angles to Pixel Coordinates
    # thetas is in range [-pi, pi]. 
    # Center-Forward Fix: 
    # At theta = 0 (straight ahead), u should be PANO_W / 2.
    # We divide by the horizontal field of view.
    u = (thetas / np.radians(FOV_H)) * PANO_W + (PANO_W / 2)
    
    # Vertical Mapping: Center the vertical FOV
    v = (PANO_H / 2) - (phis / np.radians(FOV_V)) * PANO_H
    
    return np.vstack((u, v)).T
    

def run_panoramic_fusion_loop():
    print("Initializing Data Loader & Segformer Engine...")
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    
    sample_tokens = list(loader.samples.keys())
    print(f"Total Samples to Process: {len(sample_tokens)}")

    for idx, token in enumerate(sample_tokens):
        print(f"[{idx+1}/{len(sample_tokens)}] Generating Panorama for Token: {token[:8]}...")
        
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        if not lidar_meta: continue
            
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        num_pts = raw_lidar.shape[0]
        
        # In-memory storage for point attributes
        point_colors = np.zeros((num_pts, 3), dtype=np.float16) # Store as float16 for intensity math
        point_assigned = np.zeros(num_pts, dtype=bool)
        point_depths = np.zeros(num_pts, dtype=np.float32)

        # 1. Gather colors from all Front Hemisphere cameras
        for cam_name in CAMERAS:
            cam_meta = loader.get_data(token, cam_name)
            if not cam_meta: continue
                
            img = cv2.imread(cam_meta['path'])
            h, w = img.shape[:2]
            
            # Semantic Segmentation
            logits, _ = trt_model.infer(img)
            seg_mask = np.argmax(logits, axis=1)[0]
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Project LiDAR onto this camera
            pixels, depth_mask, depths = project_points(raw_lidar, lidar_meta, cam_meta)
            
            # Identify points inside this camera's Frustum
            valid_fov = (depth_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                                     (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
            
            v_idx = np.where(valid_fov)[0]
            if len(v_idx) == 0: continue
            
            u_img, v_img = pixels[v_idx, 0].astype(int), pixels[v_idx, 1].astype(int)
            
            # Map semantic classes to colors
            classes = seg_mask[v_img, u_img]
            point_colors[v_idx] = CITYSCAPES_COLORS[classes]
            point_assigned[v_idx] = True
            point_depths[v_idx] = depths[v_idx]

        # 2. Render the Virtual Panorama
        pano_canvas = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
        
        # Only render points that were seen by at least one camera
        assigned_mask = point_assigned
        if not np.any(assigned_mask): continue
            
        final_pts_3d = raw_lidar[assigned_mask]
        final_colors = point_colors[assigned_mask]
        final_depths = point_depths[assigned_mask]
        
        # Convert 3D assigned points to Cylindrical Panorama coordinates
        pano_coords = get_cylindrical_pixels(final_pts_3d)
        
        # Draw points with Depth-Intensity Scaling
        for i in range(len(pano_coords)):
            u, v = int(pano_coords[i, 0]), int(pano_coords[i, 1])
            
            if 0 <= u < PANO_W and 0 <= v < PANO_H:
                # Calculate intensity based on depth (max 50 meters)
                dist = final_depths[i]
                intensity = np.clip(1.0 - (dist / 50.0), 0.2, 1.0)
                
                # Apply intensity to the BGR color
                pixel_color = (final_colors[i] * intensity).astype(np.uint8)
                
                # Draw the point (Radius 2 for visibility)
                cv2.circle(pano_canvas, (u, v), 2, pixel_color.tolist(), -1)

        # 3. Add HUD/Overlay Info
        cv2.putText(pano_canvas, f"Virtual Panoramic Fusion | 180 deg | {token[:8]}", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out_name = os.path.join(OUTPUT_DIR, f"pano_{idx:03d}_{token[:6]}.jpg")
        cv2.imwrite(out_name, pano_canvas)

    print(f"\n✅ All panoramic fusions generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_panoramic_fusion_loop()

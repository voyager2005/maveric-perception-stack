import os
import json
import numpy as np
import cv2
import glob
from pyquaternion import Quaternion

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
OUTPUT_DIR = "full_360_fusion_results"

# 360 Panorama Resolution
PANO_W = 2048 # Increased for 360
PANO_H = 512
FOV_V = 60    # Vertical Field of View (+/- 30 degrees)

os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

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
    # Lidar -> Ego -> Global -> Cam Ego -> Cam Sensor
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

def get_360_cylindrical_pixels(points_3d):
    """
    X (Forward) is 0 degrees -> Map to Center of image
    Y (Left) is 90 degrees -> Map to Left half
    -Y (Right) is -90 degrees -> Map to Right half
    """
    # Angle around Z axis. atan2(y, x)
    thetas = np.arctan2(points_3d[:, 1], points_3d[:, 0])
    
    # Pitch Angle
    dist_xy = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    phis = np.arctan2(points_3d[:, 2], dist_xy)
    
    # Map theta (-pi to pi) to U (0 to PANO_W)
    # thetas = 0 (Center) -> u = PANO_W / 2
    # thetas = pi (Back) -> u = 0
    # thetas = -pi (Back) -> u = PANO_W
    # We want X-Forward at Center, so we shift theta
    u = ((-thetas) / (2 * np.pi)) * PANO_W + (PANO_W / 2)
    
    # Wrap u around if it goes out of bounds
    u = np.mod(u, PANO_W)
    
    # Map phis to V
    v = (PANO_H / 2) - (phis / np.radians(FOV_V)) * PANO_H
    
    return np.vstack((u, v)).T

def run_360_fusion():
    print("Initializing 360 Fusion...")
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    sample_tokens = list(loader.samples.keys())[:10] # Process 10 frames

    for idx, token in enumerate(sample_tokens):
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        if not lidar_meta: continue
            
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        num_pts = raw_lidar.shape[0]
        
        # Point metadata
        point_colors = np.zeros((num_pts, 3), dtype=np.uint8)
        point_assigned = np.zeros(num_pts, dtype=bool)
        point_depths = np.linalg.norm(raw_lidar, axis=1)

        # 1. Background: Calculate all 360 positions and set default color
        pano_coords = get_360_cylindrical_pixels(raw_lidar)
        
        # 2. Semantic Pass: Paint points seen by the 3 Front Cameras
        for cam_name in CAMERAS:
            cam_meta = loader.get_data(token, cam_name)
            if not cam_meta: continue
            
            img = cv2.imread(cam_meta['path'])
            h, w = img.shape[:2]
            logits, _ = trt_model.infer(img)
            seg_mask = np.argmax(logits, axis=1)[0]
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            pixels, depth_mask, _ = project_points(raw_lidar, lidar_meta, cam_meta)
            valid_fov = (depth_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                                     (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
            
            v_idx = np.where(valid_fov)[0]
            if len(v_idx) == 0: continue
            
            u_img, v_img = pixels[v_idx, 0].astype(int), pixels[v_idx, 1].astype(int)
            classes = seg_mask[v_img, u_img]
            point_colors[v_idx] = CITYSCAPES_COLORS[classes]
            point_assigned[v_idx] = True

        # 3. Render Canvas
        canvas = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
        
        for i in range(num_pts):
            u, v = int(pano_coords[i, 0]), int(pano_coords[i, 1])
            if 0 <= u < PANO_W and 0 <= v < PANO_H:
                dist = point_depths[i]
                if dist > 60: continue # Clip distant points
                
                if point_assigned[i]:
                    # Painted Point: Semantic Color + Distance Brightness
                    brightness = np.clip(1.2 - (dist / 50.0), 0.3, 1.0)
                    color = (point_colors[i] * brightness).astype(np.uint8)
                    cv2.circle(canvas, (u, v), 2, color.tolist(), -1)
                else:
                    # Unpainted Background Point: Dark Grey
                    grey_val = int(np.clip(80 - (dist * 1.2), 10, 80))
                    canvas[v, u] = [grey_val, grey_val, grey_val]

        # Draw HUD lines for 0, 90, 180, 270 degrees
        for angle_x in [0, PANO_W//4, PANO_W//2, 3*PANO_W//4]:
            cv2.line(canvas, (angle_x, 0), (angle_x, 20), (255, 255, 255), 1)

        cv2.putText(canvas, "360 Lidar View | FRONT 3-CAM PAINTED", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        out_path = os.path.join(OUTPUT_DIR, f"360_fusion_{idx:03d}.jpg")
        cv2.imwrite(out_path, canvas)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    run_360_fusion()

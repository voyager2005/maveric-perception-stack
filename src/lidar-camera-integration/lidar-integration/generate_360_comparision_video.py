import os
import json
import numpy as np
import cv2
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
OUTPUT_VIDEO = "slow_mo_fusion_check.mp4"

# Dimensions
PANO_W, PANO_H = 1024, 1024
CAM_DISPLAY_SIZE = 1024
FPS = 2  # Slowed down for visual inspection

# Front Hemisphere cameras
PAINT_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

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
    # atan2 gives 0 for X (forward). 
    # thetas goes from -pi (back-right) to pi (back-left)
    thetas = np.arctan2(points_3d[:, 1], points_3d[:, 0])
    dist_xy = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    phis = np.arctan2(points_3d[:, 2], dist_xy)
    
    # -theta because we want rotation direction to match image order
    u = ((-thetas) / (2 * np.pi)) * PANO_W + (PANO_W / 2)
    u = np.mod(u, PANO_W)
    
    # 90 degree vertical sweep
    v = (PANO_H / 2) - (phis / np.radians(90)) * PANO_H
    return np.vstack((u, v)).T

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
            point_colors[v_idx] = CITYSCAPES_COLORS[seg_mask[v_img, u_img]]
            point_assigned[v_idx] = True

        # 2. Render Left Pane (Lidar)
        pano_pane = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
        pano_coords = get_360_cylindrical_pixels(raw_lidar)
        
        for i in range(num_pts):
            u, v = int(pano_coords[i, 0]), int(pano_coords[i, 1])
            if 0 <= u < PANO_W and 0 <= v < PANO_H:
                dist = point_depths[i]
                if dist > 80: continue
                
                if point_assigned[i]:
                    # Paint color with distance falloff
                    intensity = np.clip(1.3 - (dist / 50.0), 0.4, 1.0)
                    color = (point_colors[i] * intensity).astype(np.uint8)
                    cv2.circle(pano_pane, (u, v), 2, color.tolist(), -1)
                else:
                    # Grey contextual structure
                    val = int(np.clip(60 - dist, 10, 60))
                    pano_pane[v, u] = [val, val, val]

        # 3. Add Sector Overlays (6 sections)
        sectors = ["BACK", "BACK LEFT", "FRONT LEFT", "FRONT", "FRONT RIGHT", "BACK RIGHT", "BACK"]
        for i, name in enumerate(sectors):
            x = int((i / 6.0) * PANO_W)
            cv2.line(pano_pane, (x, 0), (x, PANO_H), (100, 100, 100), 1)
            if i < 6:
                cv2.putText(pano_pane, name, (x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Render Right Pane (Camera)
        driver_img = cv2.imread(cam_front_meta['path'])
        driver_img = cv2.resize(driver_img, (CAM_DISPLAY_SIZE, CAM_DISPLAY_SIZE))
        
        # 5. Combine and Finish
        final_frame = np.hstack((pano_pane, driver_img))
        cv2.putText(final_frame, f"Analysis: {token[:8]} | FRONT-PAINTED ONLY", (50, 950), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out_video.write(final_frame)
        if (idx+1) % 5 == 0: print(f"Encoded {idx+1} frames...")

    out_video.release()
    print(f"\n✅ Analysis video saved: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_visual_analysis()

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

# Video Outputs
VIDEO_1_NAME = "panoramic_lidar_180.mp4"
VIDEO_2_NAME = "three_pane_camera_fusion.mp4"

FPS = 2 # Slowed down for analysis
FRAME_LIMIT = 50 # Number of frames to process

# Front Hemisphere cameras
CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

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

def get_180_cylindrical_pixels(points_3d, w, h):
    thetas = np.arctan2(points_3d[:, 1], points_3d[:, 0])
    dist_xy = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    phis = np.arctan2(points_3d[:, 2], dist_xy)
    
    # Clip to +/- 90 degrees (180 total)
    valid_mask = (thetas >= -np.pi/2) & (thetas <= np.pi/2)
    
    # Map theta to width (u=0 is right +90, u=w is left -90)
    u = ((-thetas + np.pi/2) / np.pi) * w
    v = (h / 2) - (phis / np.radians(60)) * h
    return np.vstack((u, v)).T, valid_mask

def run_multi_video_gen():
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    sample_tokens = list(loader.samples.keys())[:FRAME_LIMIT]

    # Video 1 Setup (180 Lidar)
    v1_w, v1_h = 1500, 800
    out_v1 = cv2.VideoWriter(VIDEO_1_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (v1_w, v1_h))
    
    # Video 2 Setup (3-Pane Camera)
    pane_w, pane_h = 640, 480
    out_v2 = cv2.VideoWriter(VIDEO_2_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (pane_w * 3, pane_h))

    print(f"Starting Video Generation ({FRAME_LIMIT} frames)...")

    for idx, token in enumerate(sample_tokens):
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        if not lidar_meta: continue
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        
        # Temp storage for the current frame
        processed_panes = []
        global_painted_colors = np.zeros((len(raw_lidar), 3), dtype=np.uint8)
        global_painted_mask = np.zeros(len(raw_lidar), dtype=bool)
        global_depths = np.zeros(len(raw_lidar), dtype=np.float32)

        for cam_name in CAM_NAMES:
            cam_meta = loader.get_data(token, cam_name)
            img = cv2.imread(cam_meta['path'])
            img_h, img_w = img.shape[:2]
            
            # Segformer Inference
            logits, _ = trt_model.infer(img)
            seg_mask = np.argmax(logits, axis=1)[0]
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            
            # Opaque overlay for Video 2
            overlay = cv2.addWeighted(img, 0.6, CITYSCAPES_COLORS[seg_mask], 0.4, 0)
            
            # Projection
            pixels, mask, depths = project_points(raw_lidar, lidar_meta, cam_meta)
            fov_mask = mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
            
            # Draw Points on Pane for Video 2
            v_idx = np.where(fov_mask)[0]
            for i in v_idx:
                u, v = int(pixels[i, 0]), int(pixels[i, 1])
                cls = seg_mask[v, u]
                color = CITYSCAPES_COLORS[cls].tolist()
                cv2.circle(overlay, (u, v), 2, color, -1)
                
                # Update global painting for Video 1
                global_painted_colors[i] = color
                global_painted_mask[i] = True
                global_depths[i] = depths[i]

            pane = cv2.resize(overlay, (pane_w, pane_h))
            cv2.putText(pane, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed_panes.append(pane)

        # Write Video 2
        out_v2.write(cv2.hconcat(processed_panes))

        # Render Video 1 (180 Panoramic)
        v1_frame = np.zeros((v1_h, v1_w, 3), dtype=np.uint8)
        pano_pixels, valid_180 = get_180_cylindrical_pixels(raw_lidar, v1_w, v1_h)
        
        for i in range(len(raw_lidar)):
            if not valid_180[i]: continue
            u, v = int(pano_pixels[i, 0]), int(pano_pixels[i, 1])
            if 0 <= u < v1_w and 0 <= v < v1_h:
                dist = np.linalg.norm(raw_lidar[i])
                if global_painted_mask[i]:
                    intensity = np.clip(1.2 - (dist / 60.0), 0.3, 1.0)
                    cv2.circle(v1_frame, (u, v), 2, (global_painted_colors[i] * intensity).astype(np.uint8).tolist(), -1)
                else:
                    # Contextual structural points (Front but not in cam FOV)
                    val = int(np.clip(50 - dist, 10, 50))
                    v1_frame[v, u] = [val, val, val]

        # Labels for Video 1
        cv2.putText(v1_frame, "FRONT LEFT | FRONT | FRONT RIGHT (180 deg)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out_v1.write(v1_frame)

        if (idx + 1) % 5 == 0: print(f"Processed {idx + 1} frames...")

    out_v1.release()
    out_v2.release()
    print(f"\n✅ Videos generated:\n1. {VIDEO_1_NAME}\n2. {VIDEO_2_NAME}")

if __name__ == "__main__":
    run_multi_video_gen()

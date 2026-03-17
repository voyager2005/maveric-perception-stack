import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion

# Import your TensorRT Segformer wrapper
from trt_segformer import SegFormerTRT

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini" 
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
ENGINE_PATH = "./segformer_b1_python_build.engine"
OUTPUT_DIR = "panoramic_fusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cameras ordered for a logical 180-degree stitch
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
    # Transforms
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

def process_frame(token, loader, trt_model):
    cam_results = []
    lidar_meta = loader.get_data(token, "LIDAR_TOP")
    raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]

    for cam_name in CAMERAS:
        cam_meta = loader.get_data(token, cam_name)
        img = cv2.imread(cam_meta['path'])
        h, w = img.shape[:2]
        
        # Inference & Segmentation
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Opaque overlay
        color_mask = CITYSCAPES_COLORS[seg_mask]
        blended = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        
        # Lidar Projection
        pixels, depth_mask = project_points(raw_lidar, lidar_meta, cam_meta)
        valid_idx = np.where(depth_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & 
                             (pixels[:, 1] >= 0) & (pixels[:, 1] < h))[0]
        
        for idx in valid_idx:
            u, v = int(pixels[idx, 0]), int(pixels[idx, 1])
            class_id = seg_mask[v, u]
            p_color = CITYSCAPES_COLORS[class_id].tolist()
            cv2.circle(blended, (u, v), 2, p_color, -1)

        # Label the camera
        cv2.putText(blended, cam_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cam_results.append(blended)

    # Stitch cameras horizontally
    return cv2.hconcat(cam_results)

if __name__ == "__main__":
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    tokens = list(loader.samples.keys())[:5]

    for i, token in enumerate(tokens):
        print(f"Stitching Panoramic Sweep {i+1}...")
        panorama = process_frame(token, loader, trt_model)
        out_path = os.path.join(OUTPUT_DIR, f"panoramic_{i+1}.jpg")
        cv2.imwrite(out_path, panorama)
    
    print(f"\n✅ Panoramic visualizations saved to {OUTPUT_DIR}")

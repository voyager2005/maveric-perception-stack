import os
import json
import numpy as np
import cv2
import time
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN

# Ensure trt_segformer.py is available
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
NUM_TEST_FRAMES = 50 
WARMUP_FRAMES = 5

CAMERAS = ["CAM_FRONT"] # Benchmarking on the primary front cam

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
    return points[points[:, 2] > -1.4]

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
    if W_3d < class_rules["min_w"] or W_3d > class_rules["max_w"]: return None
    if H_3d < class_rules["min_h"] or H_3d > class_rules["max_h"]: return None
    return True # Benchmark only needs to know if a box was found

def run_benchmark():
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    trt_model = SegFormerTRT(ENGINE_PATH)
    tokens = list(loader.samples.keys())[:NUM_TEST_FRAMES + WARMUP_FRAMES]
    
    # Timers
    t_io, t_infer, t_paint, t_detect = [], [], [], []

    print(f"🚀 Starting Benchmark: {NUM_TEST_FRAMES} frames...")
    
    for i, token in enumerate(tokens):
        is_warmup = i < WARMUP_FRAMES
        
        # 1. I/O & Prep
        start = time.perf_counter()
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        cam_meta = loader.get_data(token, "CAM_FRONT")
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        img = cv2.imread(cam_meta['path'])
        prep_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_io.append(prep_time)

        # 2. Inference
        start = time.perf_counter()
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (1600, 900), interpolation=cv2.INTER_NEAREST)
        infer_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_infer.append(infer_time)

        # 3. Painting / Projection
        start = time.perf_counter()
        filtered_lidar = remove_ground(raw_lidar)
        pixels, depth_mask, depths = project_points(filtered_lidar, lidar_meta, cam_meta)
        paint_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_paint.append(paint_time)

        # 4. Detection / Clustering
        start = time.perf_counter()
        K = np.array(cam_meta['cs']['camera_intrinsic'])
        for class_id, rules in DYNAMIC_CLASSES.items():
            binary_mask = np.uint8(seg_mask == class_id)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            for j in range(1, num_labels):
                u_min, v_min, w_2d, h_2d, area = stats[j]
                if area < 100: continue
                in_box = (depth_mask) & (pixels[:, 0] >= u_min) & (pixels[:, 0] <= u_min + w_2d) & \
                                        (pixels[:, 1] >= v_min) & (pixels[:, 1] <= v_min + h_2d)
                pts_inside = np.where(in_box)[0]
                if len(pts_inside) > 0:
                    d_clust = DBSCAN(eps=1.5, min_samples=1).fit(depths[pts_inside].reshape(-1, 1))
                    build_3d_box_from_2d(centroids[j][0], centroids[j][1], w_2d, h_2d, 
                                         np.median(depths[pts_inside]), len(pts_inside), K, rules)
        detect_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_detect.append(detect_time)

        if not is_warmup and (i - WARMUP_FRAMES + 1) % 10 == 0:
            print(f"  Processed {i - WARMUP_FRAMES + 1}/{NUM_TEST_FRAMES}...")

    # Statistics Summary
    avg_total = np.mean(t_io) + np.mean(t_infer) + np.mean(t_paint) + np.mean(t_detect)
    
    print("\n" + "="*40)
    print("📊 PERCEPTION BENCHMARK RESULTS (v1)")
    print("="*40)
    print(f"{'Phase':<15} | {'Avg (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 40)
    print(f"{'1. I/O & Prep':<15} | {np.mean(t_io):>8.2f} | {np.percentile(t_io, 95):>8.2f}")
    print(f"{'2. Inference':<15} | {np.mean(t_infer):>8.2f} | {np.percentile(t_infer, 95):>8.2f}")
    print(f"{'3. Painting':<15} | {np.mean(t_paint):>8.2f} | {np.percentile(t_paint, 95):>8.2f}")
    print(f"{'4. Detection':<15} | {np.mean(t_detect):>8.2f} | {np.percentile(t_detect, 95):>8.2f}")
    print("-" * 40)
    print(f"{'TOTAL LATENCY':<15} | {avg_total:>8.2f} ms")
    print(f"{'ESTIMATED FPS':<15} | {1000/avg_total:>8.2f} Hz")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()

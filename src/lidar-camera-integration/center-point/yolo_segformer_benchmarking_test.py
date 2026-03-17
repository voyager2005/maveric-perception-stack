import os
import json
import numpy as np
import cv2
import time
from pyquaternion import Quaternion

# Import both TensorRT wrappers
try:
    from trt_segformer import SegFormerTRT
    from trt_yolo import YoloTRT
except ImportError:
    print("❌ Error: TensorRT wrappers not found.")

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini"
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
SEG_ENGINE_PATH = "./segformer_b1_python_build.engine"
YOLO_ENGINE_PATH = "./yolov8n.engine"

NUM_TEST_FRAMES = 50 
WARMUP_FRAMES = 5

CAMERAS = ["CAM_FRONT"] 

YOLO_TO_CITYSCAPES = {0: 11, 1: 18, 2: 13, 3: 17, 5: 15, 7: 14}

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

def run_benchmark():
    print("Loading TensorRT Engines...")
    yolo_model = YoloTRT(YOLO_ENGINE_PATH) 
    trt_model = SegFormerTRT(SEG_ENGINE_PATH)
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    
    tokens = list(loader.samples.keys())[:NUM_TEST_FRAMES + WARMUP_FRAMES]
    
    # Timers
    t_io, t_seg, t_yolo, t_proj, t_fuse = [], [], [], [], []

    print(f"🚀 Starting Benchmark: {NUM_TEST_FRAMES} frames...")
    
    for i, token in enumerate(tokens):
        is_warmup = i < WARMUP_FRAMES
        
        # ---------------------------------------------------------
        # 1. I/O & Prep
        # ---------------------------------------------------------
        start = time.perf_counter()
        lidar_meta = loader.get_data(token, "LIDAR_TOP")
        cam_meta = loader.get_data(token, "CAM_FRONT")
        raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
        img = cv2.imread(cam_meta['path'])
        io_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_io.append(io_time)

        h, w = img.shape[:2]

        # ---------------------------------------------------------
        # 2. SegFormer Inference
        # ---------------------------------------------------------
        start = time.perf_counter()
        logits, _ = trt_model.infer(img)
        seg_mask = np.argmax(logits, axis=1)[0]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        seg_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_seg.append(seg_time)

        # ---------------------------------------------------------
        # 3. YOLO Inference
        # ---------------------------------------------------------
        start = time.perf_counter()
        yolo_boxes, yolo_classes = yolo_model.infer(img)
        yolo_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_yolo.append(yolo_time)

        # ---------------------------------------------------------
        # 4. LiDAR Projection
        # ---------------------------------------------------------
        start = time.perf_counter()
        filtered_lidar = remove_ground(raw_lidar)
        pixels, depth_mask, depths = project_points(filtered_lidar, lidar_meta, cam_meta)
        proj_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_proj.append(proj_time)

        # ---------------------------------------------------------
        # 5. Double-Filter Fusion Logic
        # ---------------------------------------------------------
        start = time.perf_counter()
        valid_objects = 0
        for box, cls_id in zip(yolo_boxes, yolo_classes):
            yolo_cls = int(cls_id)
            if yolo_cls not in YOLO_TO_CITYSCAPES: continue
            
            cs_class = YOLO_TO_CITYSCAPES[yolo_cls]
            x1, y1, x2, y2 = box
            
            in_box = (depth_mask) & (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & \
                                    (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
            pts_in_box = np.where(in_box)[0]
            
            if len(pts_in_box) > 0:
                semantic_labels = seg_mask[pixels[pts_in_box, 1].astype(int), pixels[pts_in_box, 0].astype(int)]
                valid_pts = pts_in_box[semantic_labels == cs_class]
                
                if len(valid_pts) > 0:
                    median_depth = np.median(depths[valid_pts])
                    valid_objects += 1
                    
        fuse_time = (time.perf_counter() - start) * 1000
        if not is_warmup: t_fuse.append(fuse_time)

        if not is_warmup and (i - WARMUP_FRAMES + 1) % 10 == 0:
            print(f"  Processed {i - WARMUP_FRAMES + 1}/{NUM_TEST_FRAMES}...")

    # Statistics Summary
    avg_total = np.mean(t_io) + np.mean(t_seg) + np.mean(t_yolo) + np.mean(t_proj) + np.mean(t_fuse)
    
    print("\n" + "="*45)
    print("📊 DUAL TRT PIPELINE BENCHMARK RESULTS")
    print("="*45)
    print(f"{'Phase':<20} | {'Avg (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 45)
    print(f"{'1. I/O & Prep':<20} | {np.mean(t_io):>8.2f} | {np.percentile(t_io, 95):>8.2f}")
    print(f"{'2. SegFormer TRT':<20} | {np.mean(t_seg):>8.2f} | {np.percentile(t_seg, 95):>8.2f}")
    print(f"{'3. YOLO TRT':<20} | {np.mean(t_yolo):>8.2f} | {np.percentile(t_yolo, 95):>8.2f}")
    print(f"{'4. LiDAR Projection':<20} | {np.mean(t_proj):>8.2f} | {np.percentile(t_proj, 95):>8.2f}")
    print(f"{'5. Fusion Math':<20} | {np.mean(t_fuse):>8.2f} | {np.percentile(t_fuse, 95):>8.2f}")
    print("-" * 45)
    print(f"{'TOTAL PIPELINE LATENCY':<20} | {avg_total:>8.2f} ms")
    print(f"{'ESTIMATED FPS':<20} | {1000/avg_total:>8.2f} Hz")
    print("="*45)

if __name__ == "__main__":
    run_benchmark()

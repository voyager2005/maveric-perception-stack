import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion

# Import BOTH of our TensorRT engines!
try:
    from trt_segformer import SegFormerTRT
except ImportError:
    print("❌ Error: trt_segformer.py not found.")

try:
    from trt_yolo import YoloTRT
except ImportError:
    print("❌ Error: trt_yolo.py not found.")

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
DATA_DIR = r"/home/cv/Documents/perception/dependencies/Data/v1.0-mini"
JSON_DIR = os.path.join(DATA_DIR, "v1.0-mini")
SEG_ENGINE_PATH = "./segformer_b1_python_build.engine"
YOLO_ENGINE_PATH = "./yolov8n.engine"
OUTPUT_DIR = "yolo_trt_depth_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_TO_CITYSCAPES = {
    0: 11, # person 
    1: 18, # bicycle 
    2: 13, # car 
    3: 17, # motorcycle 
    5: 15, # bus 
    7: 14  # truck 
}

CLASS_NAMES = {
    11: "person", 13: "car", 14: "truck", 
    15: "bus", 17: "motorcycle", 18: "bicycle"
}

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

def process_yolo_trt_frame(token, loader, yolo_model, trt_model):
    cam_name = "CAM_FRONT"
    lidar_meta = loader.get_data(token, "LIDAR_TOP")
    cam_meta = loader.get_data(token, cam_name)
    
    raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
    filtered_lidar = remove_ground(raw_lidar)

    img = cv2.imread(cam_meta['path'])
    h, w = img.shape[:2]
    
    # 1. Segformer Inference (TRT)
    logits, _ = trt_model.infer(img)
    seg_mask = np.argmax(logits, axis=1)[0]
    seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. YOLO Inference (TRT)
    yolo_boxes, yolo_classes = yolo_model.infer(img)
    
    pixels, depth_mask, depths = project_points(filtered_lidar, lidar_meta, cam_meta)
    
    blended = cv2.addWeighted(img, 0.8, CITYSCAPES_COLORS[seg_mask], 0.2, 0)

    # 3. Analyze YOLO Boxes
    for box, cls_id in zip(yolo_boxes, yolo_classes):
        yolo_cls = int(cls_id)
        if yolo_cls not in YOLO_TO_CITYSCAPES:
            continue
            
        cs_class = YOLO_TO_CITYSCAPES[yolo_cls]
        class_name = CLASS_NAMES[cs_class]
        x1, y1, x2, y2 = box
        
        in_box = (depth_mask) & (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & \
                                (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
        pts_in_box = np.where(in_box)[0]
        
        if len(pts_in_box) > 0:
            u_ins = pixels[pts_in_box, 0].astype(int)
            v_ins = pixels[pts_in_box, 1].astype(int)
            semantic_labels = seg_mask[v_ins, u_ins]
            
            valid_semantic_pts = pts_in_box[semantic_labels == cs_class]
            
            if len(valid_semantic_pts) > 0:
                median_depth = np.median(depths[valid_semantic_pts])
                
                cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name.upper()} {median_depth:.1f}m"
                cv2.putText(blended, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for pt_idx in valid_semantic_pts:
                    cv2.circle(blended, (int(pixels[pt_idx, 0]), int(pixels[pt_idx, 1])), 2, (0, 0, 255), -1)
                continue 
                
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(blended, f"{class_name.upper()}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(blended, "GREEN: Depth Verified | BLUE: YOLO Only", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return blended

if __name__ == "__main__":
    print("Loading Dual TensorRT Pipeline...")
    yolo_model = YoloTRT(YOLO_ENGINE_PATH) 
    trt_model = SegFormerTRT(SEG_ENGINE_PATH)
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    
    tokens = list(loader.samples.keys())[:15]

    for i, token in enumerate(tokens):
        print(f"[{i+1}/{len(tokens)}] Processing Dual TRT Frame: {token[:8]}...")
        result_img = process_yolo_trt_frame(token, loader, yolo_model, trt_model)
        
        out_path = os.path.join(OUTPUT_DIR, f"yolo_trt_{i:03d}.jpg")
        cv2.imwrite(out_path, result_img)
    
    print(f"\n✅ Dual TRT visualizations saved to {OUTPUT_DIR}")

import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from ultralytics import YOLO

# Ensure trt_segformer is available
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
OUTPUT_DIR = "yolo_fusion_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLOv8 Nano model (will auto-download ~6MB if not present)
YOLO_MODEL_NAME = "yolov8n.pt" 

# Mapping YOLO (COCO) class IDs to SegFormer (Cityscapes) class IDs
YOLO_TO_CITYSCAPES = {
    0: 11, # person -> person
    1: 18, # bicycle -> bicycle
    2: 13, # car -> car
    3: 17, # motorcycle -> motorcycle
    5: 15, # bus -> bus
    7: 14  # truck -> truck
}

CLASS_PRIORS = {
    11: {"name": "person", "l_prior": 0.6},
    13: {"name": "car",    "l_prior": 4.5},
    14: {"name": "truck",  "l_prior": 8.0},
    15: {"name": "bus",    "l_prior": 10.0},
    17: {"name": "motorcycle", "l_prior": 2.0},
    18: {"name": "bicycle", "l_prior": 1.8}
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

def build_3d_box_from_yolo(x1, y1, x2, y2, median_depth, K, l_prior):
    """Unprojects a 2D YOLO box back into 3D using the Double-Filtered Depth."""
    fx, fy = K[0, 0], K[1, 1]
    cx_img, cy_img = K[0, 2], K[1, 2]
    
    # YOLO box center
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    w_2d = x2 - x1
    h_2d = y2 - y1

    Z = median_depth
    X = (u - cx_img) * Z / fx
    Y = (v - cy_img) * Z / fy
    
    W_3d = (w_2d * Z) / fx
    H_3d = (h_2d * Z) / fy
    L_3d = l_prior 
    
    return {"center_3d": [X, Y, Z], "dims": [W_3d, H_3d, L_3d]}

def get_8_corners_cam_frame(box_3d):
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

def process_yolo_fusion_frame(token, loader, yolo_model, trt_model):
    # We will test this on just CAM_FRONT to clearly see the pipeline logic
    cam_name = "CAM_FRONT"
    lidar_meta = loader.get_data(token, "LIDAR_TOP")
    cam_meta = loader.get_data(token, cam_name)
    
    raw_lidar = np.fromfile(lidar_meta['path'], dtype=np.float32).reshape(-1, 5)[:, :3]
    filtered_lidar = remove_ground(raw_lidar)

    img = cv2.imread(cam_meta['path'])
    h, w = img.shape[:2]
    K = np.array(cam_meta['cs']['camera_intrinsic'])
    
    # 1. Segformer Inference (Semantics)
    logits, _ = trt_model.infer(img)
    seg_mask = np.argmax(logits, axis=1)[0]
    seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. YOLO Inference (2D Geometry Proposals)
    results = yolo_model(img, verbose=False)[0]
    yolo_boxes = results.boxes.xyxy.cpu().numpy()
    yolo_classes = results.boxes.cls.cpu().numpy()
    
    # Project LiDAR
    pixels, depth_mask, depths = project_points(filtered_lidar, lidar_meta, cam_meta)
    
    # Visualization setup
    blended = cv2.addWeighted(img, 0.8, CITYSCAPES_COLORS[seg_mask], 0.2, 0)
    detected_objects = []

    # 3. The Double-Filter Logic
    for box, cls_id in zip(yolo_boxes, yolo_classes):
        yolo_cls = int(cls_id)
        if yolo_cls not in YOLO_TO_CITYSCAPES:
            continue # Ignore stop signs, dogs, etc.
            
        cs_class = YOLO_TO_CITYSCAPES[yolo_cls]
        class_name = CLASS_PRIORS[cs_class]["name"]
        
        x1, y1, x2, y2 = box
        
        # Filter 1: Is LiDAR point inside the YOLO Box?
        in_box = (depth_mask) & (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & \
                                (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
        pts_in_box = np.where(in_box)[0]
        
        if len(pts_in_box) == 0: continue
            
        # Filter 2: Does the SegFormer mask agree with YOLO?
        u_ins = pixels[pts_in_box, 0].astype(int)
        v_ins = pixels[pts_in_box, 1].astype(int)
        
        # Query the semantic mask for the points inside the YOLO box
        semantic_labels = seg_mask[v_ins, u_ins]
        
        # Only keep points where Segformer painted them as the correct class
        valid_semantic_pts = pts_in_box[semantic_labels == cs_class]
        
        # Calculate Depth using ONLY the double-verified points
        if len(valid_semantic_pts) > 0:
            median_depth = np.median(depths[valid_semantic_pts])
            
            # Physics Heuristic: Reject bad detections at extreme ranges
            if len(valid_semantic_pts) >= max(1, int(15.0 / median_depth)):
                
                box_3d = build_3d_box_from_yolo(x1, y1, x2, y2, median_depth, K, CLASS_PRIORS[cs_class]["l_prior"])
                if box_3d:
                    detected_objects.append({"class": class_name, "bbox_3d": box_3d})
                    
                    # Vis: Draw the Valid LiDAR points in Red to prove the double-filter worked
                    for pt_idx in valid_semantic_pts:
                        cv2.circle(blended, (int(pixels[pt_idx, 0]), int(pixels[pt_idx, 1])), 2, (0, 0, 255), -1)

        # Vis: Draw the raw YOLO box in Blue
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # 4. Project Final 3D Boxes
    for obj in detected_objects:
        corners_cam = get_8_corners_cam_frame(obj["bbox_3d"])
        u_c = (corners_cam[:, 0] * K[0, 0] / corners_cam[:, 2]) + K[0, 2]
        v_c = (corners_cam[:, 1] * K[1, 1] / corners_cam[:, 2]) + K[1, 2]
        corners_2d = np.vstack((u_c, v_c)).T
        
        # Draw 3D Box in Green
        draw_3d_box_edges(blended, corners_2d, color=(0, 255, 0))
        cv2.putText(blended, f"{obj['class'].upper()} {obj['bbox_3d']['center_3d'][2]:.1f}m", 
                    (int(corners_2d[0,0]), int(corners_2d[0,1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(blended, "BLUE: YOLO 2D | GREEN: DOUBLE-FILTERED 3D", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return blended

if __name__ == "__main__":
    print("Loading Models...")
    yolo_model = YOLO(YOLO_MODEL_NAME) # Will download yolov8n.pt automatically
    trt_model = SegFormerTRT(ENGINE_PATH)
    loader = MinimalDataLoader(JSON_DIR, DATA_DIR)
    
    tokens = list(loader.samples.keys())[:15]

    for i, token in enumerate(tokens):
        print(f"[{i+1}/{len(tokens)}] Processing YOLO Fusion: {token[:8]}...")
        result_img = process_yolo_fusion_frame(token, loader, yolo_model, trt_model)
        
        out_path = os.path.join(OUTPUT_DIR, f"yolo_fusion_{i:03d}.jpg")
        cv2.imwrite(out_path, result_img)
    
    print(f"\n✅ YOLO Fusion visualizations saved to {OUTPUT_DIR}")

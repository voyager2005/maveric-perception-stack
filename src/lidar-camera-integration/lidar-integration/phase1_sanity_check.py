import json
import os
import numpy as np
import cv2
from pyquaternion import Quaternion

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
BASE_DATA_DIR = "/home/cv/Documents/points/Data/v1.0-mini"
JSON_DIR = os.path.join(BASE_DATA_DIR, "v1.0-mini")

CAMERAS_TO_TEST = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
# ==========================================

class RawNuScenesParser:
    def __init__(self, json_dir):
        print(f"Loading JSON metadata from {json_dir}...")
        
        def load_json(filename):
            with open(os.path.join(json_dir, filename)) as f:
                return {item['token']: item for item in json.load(f)}

        # Load main tables
        self.samples = load_json("sample.json")
        self.sample_data = load_json("sample_data.json")
        self.calibrated_sensors = load_json("calibrated_sensor.json")
        self.ego_poses = load_json("ego_pose.json")
        
        # Load sensor.json to map sensor_token to channel name (e.g., "CAM_FRONT")
        with open(os.path.join(json_dir, "sensor.json")) as f:
            sensors_raw = json.load(f)
        self.sensor_channel_map = {s['token']: s['channel'] for s in sensors_raw}
        
        # 1. Pre-compute calibrated_sensor -> channel string
        self.calib_to_channel = {}
        for calib_token, calib_data in self.calibrated_sensors.items():
            sensor_token = calib_data['sensor_token']
            self.calib_to_channel[calib_token] = self.sensor_channel_map[sensor_token]
            
        # 2. Pre-compute sample_token -> {channel: sample_data_record}
        self.sample_data_map = {}
        for sd_token, sd in self.sample_data.items():
            s_token = sd['sample_token']
            calib_token = sd['calibrated_sensor_token']
            
            # Get the human-readable channel name (e.g., "LIDAR_TOP")
            channel = self.calib_to_channel[calib_token]
            
            if s_token not in self.sample_data_map:
                self.sample_data_map[s_token] = {}
            
            # Store the sample_data record under its specific channel
            self.sample_data_map[s_token][channel] = sd
            
        print("✅ Metadata loaded and relational mapping complete.")

    def get_sensor_data(self, sample_token, cam_name, lidar_name="LIDAR_TOP"):
        # Look up the pre-mapped records for this specific sample
        if sample_token not in self.sample_data_map:
            raise ValueError(f"Sample token {sample_token} not found in sample_data.")
            
        sample_records = self.sample_data_map[sample_token]
        
        if lidar_name not in sample_records:
            raise ValueError(f"{lidar_name} data not found for sample {sample_token}.")
        if cam_name not in sample_records:
            raise ValueError(f"{cam_name} data not found for sample {sample_token}.")
            
        # 1. Get the exact sample_data records
        lidar_sd = sample_records[lidar_name]
        cam_sd = sample_records[cam_name]
        
        # 2. Get Extrinsics & Intrinsics
        lidar_cs = self.calibrated_sensors[lidar_sd['calibrated_sensor_token']]
        cam_cs = self.calibrated_sensors[cam_sd['calibrated_sensor_token']]
        
        # 3. Get Poses in Global Map
        lidar_pose = self.ego_poses[lidar_sd['ego_pose_token']]
        cam_pose = self.ego_poses[cam_sd['ego_pose_token']]
        
        return {
            "lidar_path": os.path.join(BASE_DATA_DIR, lidar_sd['filename']),
            "cam_path": os.path.join(BASE_DATA_DIR, cam_sd['filename']),
            "lidar_cs": lidar_cs, "cam_cs": cam_cs,
            "lidar_pose": lidar_pose, "cam_pose": cam_pose
        }

def project_lidar_to_cam(points, matrices):
    """ Transforms 3D LiDAR points to 2D image pixels """
    
    # 1. LiDAR to Ego (at t_lidar)
    R_lidar_cs = Quaternion(matrices['lidar_cs']['rotation']).rotation_matrix
    T_lidar_cs = np.array(matrices['lidar_cs']['translation'])
    points_ego = np.dot(R_lidar_cs, points.T).T + T_lidar_cs
    
    # 2. Ego to Global (at t_lidar)
    R_lidar_pose = Quaternion(matrices['lidar_pose']['rotation']).rotation_matrix
    T_lidar_pose = np.array(matrices['lidar_pose']['translation'])
    points_global = np.dot(R_lidar_pose, points_ego.T).T + T_lidar_pose
    
    # 3. Global to Ego (at t_cam)
    R_cam_pose = Quaternion(matrices['cam_pose']['rotation']).rotation_matrix
    T_cam_pose = np.array(matrices['cam_pose']['translation'])
    points_ego_cam = np.dot(R_cam_pose.T, (points_global - T_cam_pose).T).T
    
    # 4. Ego to Camera (at t_cam)
    R_cam_cs = Quaternion(matrices['cam_cs']['rotation']).rotation_matrix
    T_cam_cs = np.array(matrices['cam_cs']['translation'])
    points_cam = np.dot(R_cam_cs.T, (points_ego_cam - T_cam_cs).T).T
    
    # 5. Depth Filtering (Must be in front of the camera)
    depths = points_cam[:, 2]
    mask = depths > 0.1 
    points_cam = points_cam[mask]
    depths = depths[mask]
    
    # 6. Camera Frame to Image Pixels
    K = np.array(matrices['cam_cs']['camera_intrinsic'])
    points_img = np.dot(K, points_cam.T).T
    
    u = points_img[:, 0] / points_img[:, 2]
    v = points_img[:, 1] / points_img[:, 2]
    
    pixel_coords = np.vstack((u, v)).T
    return pixel_coords, depths

def run_sanity_check():
    parser = RawNuScenesParser(JSON_DIR)
    
    # Grab the very first sample in the dataset as our test frame
    test_sample_token = list(parser.samples.keys())[0]
    print(f"\nProcessing Sample Token: {test_sample_token}")

    for cam_name in CAMERAS_TO_TEST:
        print(f"\n--- Testing {cam_name} ---")
        
        # 1. Extract all routing matrices
        data = parser.get_sensor_data(test_sample_token, cam_name)
        
        # 2. Load Image
        img = cv2.imread(data['cam_path'])
        if img is None:
            print(f"❌ Could not load image: {data['cam_path']}")
            continue
        h, w = img.shape[:2]
        
        # 3. Load LiDAR (NuScenes .bin files are float32, with 5 dimensions per point)
        # We only need the first 3 (X, Y, Z)
        lidar_bin = np.fromfile(data['lidar_path'], dtype=np.float32).reshape(-1, 5)
        points_3d = lidar_bin[:, :3] 
        print(f"Loaded {len(points_3d)} LiDAR points.")
        
        # 4. Project
        pixels, depths = project_lidar_to_cam(points_3d, data)
        
        # 5. Filter points that fall outside the image dimensions
        valid_idx = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        
        valid_pixels = pixels[valid_idx].astype(int)
        valid_depths = depths[valid_idx]
        print(f"Projected {len(valid_pixels)} points successfully onto {cam_name}.")

        # 6. Visualization (Color map based on depth: Red=Close, Blue=Far)
        max_depth = 50.0 # 50 meters
        for (u, v), depth in zip(valid_pixels, valid_depths):
            # Normalize depth for coloring
            color_ratio = min(1.0, depth / max_depth)
            color = (
                int(255 * color_ratio),       # B
                int(255 * (1 - abs(color_ratio - 0.5) * 2)), # G
                int(255 * (1 - color_ratio))  # R
            )
            cv2.circle(img, (u, v), 2, color, -1)
            
        # 7. Save the output
        output_name = f"sanity_check_{cam_name}.jpg"
        cv2.imwrite(output_name, img)
        print(f"✅ Saved visualization to {output_name}")

if __name__ == "__main__":
    run_sanity_check()

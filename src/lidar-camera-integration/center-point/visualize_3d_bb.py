import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion

# ==========================================
# 🛑 CONFIGURATION & PATHS
# ==========================================
BASE_DATA_DIR = "/home/cv/Documents/points/Data/v1.0-mini"
JSON_DIR = os.path.join(BASE_DATA_DIR, "v1.0-mini")
CAM_NAME = "CAM_FRONT"
# ==========================================

# 1. Copy over your fully working RawNuScenesParser from Phase 1
class RawNuScenesParser:
    def __init__(self, json_dir):
        def load_json(filename):
            with open(os.path.join(json_dir, filename)) as f:
                return {item['token']: item for item in json.load(f)}

        self.samples = load_json("sample.json")
        self.sample_data = load_json("sample_data.json")
        self.calibrated_sensors = load_json("calibrated_sensor.json")
        self.ego_poses = load_json("ego_pose.json")
        
        with open(os.path.join(json_dir, "sensor.json")) as f:
            sensors_raw = json.load(f)
        self.sensor_channel_map = {s['token']: s['channel'] for s in sensors_raw}
        
        self.calib_to_channel = {}
        for calib_token, calib_data in self.calibrated_sensors.items():
            self.calib_to_channel[calib_token] = self.sensor_channel_map[calib_data['sensor_token']]
            
        self.sample_data_map = {}
        for sd_token, sd in self.sample_data.items():
            s_token = sd['sample_token']
            calib_token = sd['calibrated_sensor_token']
            channel = self.calib_to_channel[calib_token]
            if s_token not in self.sample_data_map:
                self.sample_data_map[s_token] = {}
            self.sample_data_map[s_token][channel] = sd

    def get_sensor_data(self, sample_token, cam_name, lidar_name="LIDAR_TOP"):
        sample_records = self.sample_data_map[sample_token]
        lidar_sd = sample_records[lidar_name]
        cam_sd = sample_records[cam_name]
        
        return {
            "lidar_path": os.path.join(BASE_DATA_DIR, lidar_sd['filename']),
            "cam_path": os.path.join(BASE_DATA_DIR, cam_sd['filename']),
            "lidar_cs": self.calibrated_sensors[lidar_sd['calibrated_sensor_token']], 
            "cam_cs": self.calibrated_sensors[cam_sd['calibrated_sensor_token']],
            "lidar_pose": self.ego_poses[lidar_sd['ego_pose_token']], 
            "cam_pose": self.ego_poses[cam_sd['ego_pose_token']]
        }

# 2. Phase 1 Projection Math
def project_lidar_to_cam(points, matrices):
    R_lidar_cs = Quaternion(matrices['lidar_cs']['rotation']).rotation_matrix
    T_lidar_cs = np.array(matrices['lidar_cs']['translation'])
    points_ego = np.dot(R_lidar_cs, points.T).T + T_lidar_cs
    
    R_lidar_pose = Quaternion(matrices['lidar_pose']['rotation']).rotation_matrix
    T_lidar_pose = np.array(matrices['lidar_pose']['translation'])
    points_global = np.dot(R_lidar_pose, points_ego.T).T + T_lidar_pose
    
    R_cam_pose = Quaternion(matrices['cam_pose']['rotation']).rotation_matrix
    T_cam_pose = np.array(matrices['cam_pose']['translation'])
    points_ego_cam = np.dot(R_cam_pose.T, (points_global - T_cam_pose).T).T
    
    R_cam_cs = Quaternion(matrices['cam_cs']['rotation']).rotation_matrix
    T_cam_cs = np.array(matrices['cam_cs']['translation'])
    points_cam = np.dot(R_cam_cs.T, (points_ego_cam - T_cam_cs).T).T
    
    depths = points_cam[:, 2]
    depth_mask = depths > 0.1 
    z_safe = np.where(depths == 0, 1e-6, depths)
    
    K = np.array(matrices['cam_cs']['camera_intrinsic'])
    points_img = np.dot(K, points_cam.T).T
    
    u = points_img[:, 0] / z_safe
    v = points_img[:, 1] / z_safe
    
    return np.vstack((u, v)).T, depth_mask

# 3. Generating the 8 Corners of a 3D Bounding Box
def get_3d_box_corners(x, y, z, w, l, h, yaw):
    """ Converts center coordinates + dimensions into 8 corner points """
    # Create 8 corners in the object's local coordinate frame
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotate by yaw
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = np.dot(rot_mat, corners)
    
    # Translate to global/lidar frame position
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners.T # Shape: (8, 3)

def draw_3d_box(img, corners_2d, color=(0, 255, 0), thickness=2):
    """ Draws the wireframe using OpenCV """
    corners = corners_2d.astype(int)
    
    # Bottom Face (0,1,2,3)
    cv2.line(img, tuple(corners[0]), tuple(corners[1]), color, thickness)
    cv2.line(img, tuple(corners[1]), tuple(corners[2]), color, thickness)
    cv2.line(img, tuple(corners[2]), tuple(corners[3]), color, thickness)
    cv2.line(img, tuple(corners[3]), tuple(corners[0]), color, thickness)
    
    # Top Face (4,5,6,7)
    cv2.line(img, tuple(corners[4]), tuple(corners[5]), color, thickness)
    cv2.line(img, tuple(corners[5]), tuple(corners[6]), color, thickness)
    cv2.line(img, tuple(corners[6]), tuple(corners[7]), color, thickness)
    cv2.line(img, tuple(corners[7]), tuple(corners[4]), color, thickness)
    
    # Vertical Pillars connecting Top and Bottom
    cv2.line(img, tuple(corners[0]), tuple(corners[4]), color, thickness)
    cv2.line(img, tuple(corners[1]), tuple(corners[5]), color, thickness)
    cv2.line(img, tuple(corners[2]), tuple(corners[6]), color, thickness)
    cv2.line(img, tuple(corners[3]), tuple(corners[7]), color, thickness)

def run_sanity_visualization():
    parser = RawNuScenesParser(JSON_DIR)
    
    # Grab the first sample
    test_sample_token = list(parser.samples.keys())[0]
    data = parser.get_sensor_data(test_sample_token, CAM_NAME)
    
    img = cv2.imread(data['cam_path'])
    h_img, w_img = img.shape[:2]
    
    # 1. Overlay Raw LiDAR for Context
    raw_lidar = np.fromfile(data['lidar_path'], dtype=np.float32).reshape(-1, 5)
    pixels, depth_mask = project_lidar_to_cam(raw_lidar[:, :3], data)
    
    valid_idx = np.where(depth_mask & 
                         (pixels[:, 0] >= 0) & (pixels[:, 0] < w_img) & 
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < h_img))[0]
    
    for idx in valid_idx:
        u, v = int(pixels[idx, 0]), int(pixels[idx, 1])
        cv2.circle(img, (u, v), 1, (200, 200, 200), -1) # Draw faint grey dots
        
    # 2. Inject a Synthetic Bounding Box (Like a CenterPoint Output)
    # Placing a car 10 meters straight ahead (X), slightly to the left (Y), on the ground (Z)
    synthetic_box = {
        "x": 10.0, "y": 2.0, "z": -1.5, 
        "w": 1.8, "l": 4.5, "h": 1.5, 
        "yaw": 0.0
    }
    
    # 3. Calculate the 8 mathematical corners in 3D space
    corners_3d = get_3d_box_corners(
        synthetic_box["x"], synthetic_box["y"], synthetic_box["z"], 
        synthetic_box["w"], synthetic_box["l"], synthetic_box["h"], synthetic_box["yaw"]
    )
    
    # 4. Project the 8 corners onto the 2D image plane
    corners_2d, corners_mask = project_lidar_to_cam(corners_3d, data)
    
    # 5. Draw it if all 8 corners are in front of the camera
    if np.all(corners_mask):
        draw_3d_box(img, corners_2d, color=(0, 255, 0), thickness=3)
        print("✅ Successfully projected 3D Bounding Box onto image.")
    else:
        print("⚠️ Box is behind the camera or out of bounds.")
        
    # Save the result
    output_name = "bb_projection_sanity_check.jpg"
    cv2.imwrite(output_name, img)
    print(f"Saved visualization to {output_name}")

if __name__ == "__main__":
    run_sanity_visualization()

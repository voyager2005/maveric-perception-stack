import cv2
import numpy as np
import matplotlib.pyplot as plt
from trt_segformer import SegFormerTRT # Your custom wrapper
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

# ==========================================
# 🛑 CONFIGURATION SECTION (FILL THESE!)
# ==========================================
BAG_PATH = "my_test_data.mcap" # Path to the bag file from Robotics Team
ENGINE_PATH = "segformer_b1_python_build.engine"

# TOPIC NAMES (Ask Robotics Team for exact names)
IMAGE_TOPIC = "/camera/image_raw"
LIDAR_TOPIC = "/scan_cloud" # If 2D lidar, ask them to remap /scan -> PointCloud2

# 🛑 CALIBRATION MATRICES (The "Golden" Numbers)
# REPLACE these with the data from the Robotics Team
# 1. Camera Intrinsic (K) - 3x3
K = np.array([
    [800.0, 0.0, 640.0],
    [0.0, 800.0, 360.0],
    [0.0, 0.0, 1.0]
])

# 2. Distortion Coefficients (D) - [k1, k2, p1, p2, k3]
D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 3. Extrinsic: LiDAR -> Camera Transform (4x4)
# This moves a point from LiDAR Frame to Camera Frame
T_LIDAR_CAM = np.array([
    [0, -1, 0, 0.0],  # Example: 90 deg rotation
    [0, 0, -1, 0.0],
    [1, 0, 0, 0.2],   # Example: 20cm forward
    [0, 0, 0, 1.0]
])
# ==========================================

def get_calibration_status(bag_path):
    print(f"Opening Bag: {bag_path}")
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    
    # Create filters to read only what we need
    reader.set_filter(StorageFilter(topics=[IMAGE_TOPIC, LIDAR_TOPIC]))

    bridge = CvBridge()
    trt_seg = SegFormerTRT(ENGINE_PATH)
    
    last_image = None
    last_cloud = None
    
    print("Scanning bag for synchronized pairs...")
    
    # Simple loop to find a pair of messages close in time
    # In a real node, you'd use message_filters.ApproximateTimeSynchronizer
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic == IMAGE_TOPIC:
            msg = deserialize_message(data, Image)
            last_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
        elif topic == LIDAR_TOPIC:
            msg = deserialize_message(data, PointCloud2)
            # Convert ROS PointCloud2 to Numpy (x, y, z)
            gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            last_cloud = np.array(list(gen))

        # Once we have both, process a frame
        if last_image is not None and last_cloud is not None:
            process_fusion(last_image, last_cloud, trt_seg)
            
            # Reset to avoid processing same pair twice
            last_image = None 
            last_cloud = None
            
            # Stop after 1 frame for simple validation (remove break to play video)
            print("Processed one frame. Check the plot window.")
            break 

def process_fusion(image, cloud_points, model):
    # 1. Undistort Image (Crucial for SegFormer accuracy)
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    rect_img = cv2.undistort(image, K, D, None, new_K)
    
    # 2. Get Segmentation Mask
    # Returns (1, 19, H/4, W/4)
    logits, _ = model.infer(rect_img) 
    seg_mask = np.argmax(logits, axis=1)[0]
    seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Project LiDAR Points
    # A. Transform to Camera Coordinate Frame
    # Add homogeneous 1
    num_pts = cloud_points.shape[0]
    pts_hom = np.hstack((cloud_points, np.ones((num_pts, 1))))
    
    # Apply Extrinsics (LiDAR -> Cam)
    pts_cam = (T_LIDAR_CAM @ pts_hom.T).T  # Shape: (N, 4)
    
    # Filter points behind the camera (Z < 0)
    pts_cam = pts_cam[pts_cam[:, 2] > 0]
    
    # B. Project to 2D Image Plane (using new_K because image is rectified)
    # x_img = fx * (x/z) + cx
    u = (pts_cam[:, 0] * new_K[0,0] / pts_cam[:, 2]) + new_K[0,2]
    v = (pts_cam[:, 1] * new_K[1,1] / pts_cam[:, 2]) + new_K[1,2]
    
    # Filter points outside image bounds
    mask_u = (u >= 0) & (u < w)
    mask_v = (v >= 0) & (v < h)
    valid_indices = mask_u & mask_v
    
    u_valid = u[valid_indices].astype(int)
    v_valid = v[valid_indices].astype(int)
    depth_valid = pts_cam[valid_indices, 2] # Z depth
    
    # 4. Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Show Segmentation overlay
    ax.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    ax.imshow(seg_mask, alpha=0.5, cmap='tab20') # Semi-transparent mask
    
    # Plot LiDAR points
    # Color them by depth (Yellow=Close, Blue=Far)
    sc = ax.scatter(u_valid, v_valid, c=depth_valid, s=3, cmap='jet_r', vmin=0, vmax=20)
    plt.colorbar(sc, label='Depth (m)')
    
    ax.set_title("Calibration Validator: LiDAR (Dots) vs Camera (Image)")
    plt.show()

if __name__ == "__main__":
    get_calibration_status(BAG_PATH)

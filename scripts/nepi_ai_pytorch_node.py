#!/usr/bin/env python


import rospy
import torch
import threading
import copy
import numpy as np
import cv2

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridge

np.bool = np.bool_


# Define your PyTorch model and load the weights
# model = ...

class YOLOObjectDetector():
    CHECK_RATE = 10
    img_msg = None
    img_last_stamp = None
    img_lock = threading.Lock()
    depth_map_msg = None
    depth_map_last_stamp = None
    depth_map_lock = threading.Lock() 
    pc_msg = None
    pc_last_stamp = None
    pc_lock = threading.Lock()

    def __init__(self):
        # Initialize parameters and fields.
        self.init_params()
        self.init_fields()

        # Initialize publishers and subscribers.
        self.img_sub = rospy.Subscriber(self.camera_topic, Image, self.image_cb)
        self.depth_map_sub = rospy.Subscriber(self.depth_map_topic, Image, self.depth_map_cb)
        self.pc_sub = rospy.Subscriber(self.pc_topic, PointCloud2, self.pc_cb)

        # Initialize the YOLO model.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('/home/nepi/pytorch_ws/models/yolov5', \
                                    'custom', \
                                    source="local", \
                                    path="/home/nepi/pytorch_ws/models/yolov5/checkpoints/yolov5s.pt" \
                     ).to(self.device)
        self.model.conf = 0.3  # Confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.model.max_det = 1000  # Maximum number of detections per image
        self.model.eval()

        

        check_interval_sec = float(1) / self.CHECK_RATE
        rospy.Timer(rospy.Duration(check_interval_sec), self.detection_callback)

    def init_fields(self):
        # OpenCV bridge.
        self.bridge = CvBridge()
        self.window_name = 'YOLO-Object-Detector'
        
        # Image fields.
        self.img = None
        self.img_lock = threading.Lock()
        self.img_status = False
        self.img_last_stamp = None
        
        # Depth map fields.
        self.depth = None
        self.depth_lock = threading.Lock()
        self.depth_status = False
        self.depth_last_stamp = None

        # Point cloud fields.
        self.pc = None
        self.pc_lock = threading.Lock()
        self.pc_status = False
        self.pc_last_stamp = None


    def init_params(self):
        rospy.set_param('~camera_topic', '/nepi/s2x/nexigo_n60_fhd_24/idx/color_2d_image')
        rospy.set_param('~depth_map_topic', '/nepi/s2x/nexigo_n60_fhd_24/idx/depth_map')
        rospy.set_param('~pc_topic', '/nepi/s2x/nexigo_n60_fhd_24/pointcloud')

        self.camera_topic = rospy.get_param('~camera_topic')
        self.depth_map_topic = rospy.get_param('~depth_map_topic')
        self.pc_topic = rospy.get_param('~pc_topic')


    def image_cb(self, msg):
        with self.img_lock:
            if self.img is not None and msg.header.stamp == self.img_last_stamp:
                # Ignore because same msg.
                return
            
            # Convert the image message to OpenCV image.
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                rospy.logerr('CvBridge Error: {}'.format(e))
                return

            self.img = copy.deepcopy(img)
            self.img_last_stamp = msg.header.stamp

    def depth_map_cb(self, msg):
        raise NotImplementedError

    def pc_cb(self, msg):
        raise NotImplementedError
    
    def detection_callback(self, event):
        with self.img_lock:
            if self.img is None:
                # No image yet.
                return

            # Detect object in the image
            results = self.model(self.img)
            
            # Save results
            results.save(save_dir='./img/', exist_ok=True)



def main():
    rospy.loginfo('Starting Pytorch YOLO Object Detector Node')
    rospy.init_node('yolo_object_detector_node', anonymous=True)
    yolo_object_detector = YOLOObjectDetector()
    rospy.spin()

if __name__ == '__main__':
    main()

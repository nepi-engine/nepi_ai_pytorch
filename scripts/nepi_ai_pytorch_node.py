#!/usr/bin/env python


import rospy
import torch

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_msg
from nepi_edge_sdk_base import nepi_img

from sensor_msgs.msg import Image

np.bool = np.bool_


# Define your PyTorch model and load the weights
# model = ...

class PytorchDetector():

    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "pytorch_ros" # Can be overwitten by luanch command
    def __init__(self):
        #### APP NODE INIT SETUP ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.node_name = nepi_ros.get_node_name()
        self.base_namespace = nepi_ros.get_base_namespace()
        nepi_msg.createMsgPublishers(self)
        nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
        ##############################
        # Initialize parameters and fields.
        node_params = nepi_ros.get_param(self,"~")
        nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        try:
            self.node_base_namespace = nepi_ros.get_param(self,"~node_base_namespace")
            self.weights_path = nepi_ros.get_param(self,"~weights_path")
            self.config_path = nepi_ros.get_param(self,"~config_path")
            self.source_image_topic = nepi_ros.get_param(self,"~source_img_topic")
            self.threshold = nepi_ros.get_param(self,"~detector_threshold")
        except Exception as e:
            nepi_msg.publishMsgErr(self,"Failed to get required node info from param server: " + str(e) )

        try:
            model_info = nepi_ros.get_param(self,"~model")
            self.config_file_path = os.path.join(self.weight_path, model_info.weight_file.name)
            self.config_file_path = os.path.join(self.config_path, model_info.config_file.name)
            self.classes = model.detection_clases.names
        except Exception as e:
            nepi_msg.publishMsgErr(self,"Failed to get required model info from params: " + str(e) )

        # Load the model
        # Initialize the YOLO model.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('/home/nepi/pytorch_ws/models/yolov5', \
                                    'custom', \
                                    source="local", \
                                    path="/home/nepi/pytorch_ws/models/yolov5/checkpoints/yolov5s.pt" \
                     ).to(self.device)
        
        
        # Create AI Node Publishers
        SOURCE_IMAGE_PUB_TOPIC = self.node_base_namespace + "/source_image"
        self.source_image_pub = rospy.Publisher(SOURCE_IMAGE_PUB_TOPIC, Image, queue_size = 1)

        DETECTION_IMAGE_PUB_TOPIC = self.node_base_namespace + "/detection_image"
        self.detection_image_pub = rospy.Publisher(DETECTION_IMAGE_PUB_TOPIC, Image,  queue_size = 1)
        
        FOUND_OBJECT_PUB_TOPIC = self.node_base_namespace + "/found_object"
        self.found_object_pub = rospy.Publisher(FOUND_OBJECT_PUB_TOPIC, ObjectCount,  queue_size = 1)

        BOUNDING_BOXES_PUB_TOPIC = self.node_base_namespace + "/bounding_boxes"
        self.bounding_boxes_pub = rospy.Publisher(BOUNDING_BOXES_PUB_TOPIC, BoundingBoxes, queue_size = 1)

        # Create AI Node Subscribers
        THRSHOLD_SUB_TOPIC = self.node_base_namespace + '/' + self.node_name + '/set_threshold'
        self.set_threshold_sub = rospy.Subscriber(THRSHOLD_SUB_TOPIC, Float32, self.updateThresholdCb, queue_size=1)

        IMAGE_SUB_TOPIC = self.source_image_topic
        self.set_threshold_sub = rospy.Subscriber(IMAGE_SUB_TOPIC, Image, self.updateDetectionCb, queue_size=1)

        #########################################################
        ## Initiation Complete
        nepi_msg.publishMsgInfo(self,"Initialization Complete")
        # Spin forever (until object is detected)
        nepi_ros.spin()
        #########################################################        
              

    def updateThresholdCb(self,msg)
        theshold = msg.data
        if (threshold < self.MIN_THRESHOLD):
            threshold = self.MIN_THRESHOLD
        elif (threshold > self.MAX_THRESHOLD):
            threshold = self.MAX_THRESHOLD
        self.updateThreshold(threshold)


    def updateDetectionCb(self,source_img_msg):
        # Update model settings
        self.model.conf = 0.3  # Confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.model.max_det = 20  # Maximum number of detections per image
        self.model.eval()

        #  Convert image from ros to cv2
        cv2_img = nepi_img.rosimg_to_cv2img(source_img_msg)
        ros_timestamp = img_msg.header.stamp

        #  Run model against image
        results = self.model(self.img)

        detection_img_msg = source_img_msg

        self.publishImages(source_img_msg, detection_img_msg)



    def publishImages(self,ros_source_img, ros_detection_img):
        ros_timestamp = 
        if not rospy.is_shutdown():
            self.source_image_pub.publish(ros_source_img)
            self.detection_image_pub_image_pub.publish(ros_detection_img)

    def publishDetectionData(self):
        detection_dict_list = self.getDetectionData()
        found_object_msg = ObjectCount()
        found_object_msg.header.stamp = process_time
        found_object_msg.count = len(detection_dict_list)
        if not rospy.is_shutdown():
            self.found_object_pub.publish(found_object_msg)

        if len(detection_dict_list) > 0:
            bounding_box_msg_list = []
            for detection_dict in detection_dict_list:
                bounding_box_msg = BoundingBox()
                bounding_box_msg.Class = detection_dict['Class']
                bounding_box_msg.id = detection_dict['id']
                bounding_box_msg.uid = detection_dict['uid']
                bounding_box_msg.probability = detection_dict['probability']
                bounding_box_msg.xmin = detection_dict['box_xmin']
                bounding_box_msg.ymin = detection_dict['box_ymin']
                bounding_box_msg.xmax = detection_dict['box_xmax']
                bounding_box_msg.ymax = detection_dict['box_ymax']
                bounding_box_msg_list.append(bounding_box_msg)
            bounding_boxes_msg = BoundingBoxes()
            bounding_boxes_msg.header.stamp = process_time
            bounding_boxes_msg.image_header = image_header
            bounding_boxes_msg.image_topic = self.source_image_topic
            bounding_boxes_msg.bounding_boxes = bounding_box_msg_list
            if not rospy.is_shutdown():
                self.bounding_boxes_pub.publish(bounding_boxes_msg)



    def get_classes_color_list(self,classes_str_list):
        rgb_list = []
        if len(classes_str_list) > 0:
            cmap = plt.get_cmap('viridis')
            color_list = cmap(np.linspace(0, 1, len(classes_str_list))).tolist()
            for color in color_list:
                for i in range(3):
                    rgb.append(int(color[i]*255))
                rgb_list.append(rgb)
        return rgb_list
                


if __name__ == '__main__':
    PytorchDetector()

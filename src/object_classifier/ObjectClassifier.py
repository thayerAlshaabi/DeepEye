# coding: utf-8
"""
This class is a deep convolutional neural network implemented using Tensorflow Object Detection API.
The API has models trained on the COCO dataset and KITTI dataset.
For more information about the API please refer to
    - Docs: https://github.com/tensorflow/models/tree/master/research/object_detection
    - COCO Dataset: http://cocodataset.org/#home
    - KITTI Dataset: http://www.cvlibs.net/datasets/kitti/

We will be using this API for three purposes:
    - detecting relevant objects,
    - creating bounding boxes, and
    - classifying the objects (car, pedestrian, motorcycle, etc.)

----------------------

# Licensing Information:
The following code was adapted from the Tensorflow Object Detection API
licensed under Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)

# Original SourceCode:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import numpy as np
import tensorflow as tf
import cv2
import os, sys
import six.moves.urllib as urllib
import tarfile, zipfile
import mss
import mss.tools
import time

# This is needed for relative paths since the code is stored in the object_detection folder.
sys.path.append("..")

# imports from the object detection module.
from object_classifier.object_detection.utils import label_map_util, visualization_utils
# ---------------------------------------------------------------------------- #

class ObjectClassifier:
    """
    Note: The constructor doesn't necessarily require passing any parameters as
    they all have some satisfactory default values to start with.

    # ------------------------------------------------------------------------ #

    For Advacned Customization

    - Model Architecture:
        The API lists many different pre-trained models of the state-of-the-art CNN architectures
        including SSD, Inception, Resnet, R-CNN, NAS, etc...
        By default, our implementation uses the (faster_rcnn_nas | mAP=43) model.
        It's relatively slow but has one of the best mean average precision (mAP) as of today's standards.

        Note: mAP is is the product of precision and recall on detecting bounding boxes.
        In short, the higher the mAP score, the more accurate the network is
        but that comes at the cost of processing speed.

        However, you can pass in a different (classifier_codename, dataset_codename), and our class
        would download any desired model from the api, if it hasn't been already downloaded to your device.
        You could also change the (classifier_threshold), which would limit the displayed detections based on their detection scores.

        A list of pre-trained models could be found here:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    """
    # Constructor
    def __init__(self,
        classifier_codename = 'faster_rcnn_nas_coco_2017_11_08',
        dataset_codename = 'mscoco',
        classifier_threshold = .75,
        visualization = False,
        diagnostic_mode = False):

        # Boolean flag for visualization utils
        self.visualization = visualization
        self.diagnostic_mode = diagnostic_mode
        
        # Folder name for object detection module
        self.FOLDER_NAME = 'object_detection'

        # Compressed file for the pre-trained model
        self.MODEL_FILE = classifier_codename + '.tar.gz'

        # Relative path for the MODEL FOLDER
        self.MODEL_FOLDER_PATH = os.path.join(os.getcwd(), 'object_classifier', self.FOLDER_NAME)

        # Relative path for the MODEL FILE
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_FOLDER_PATH, self.MODEL_FILE)

        # URL for TF Object Detection API
        self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = os.path.join(self.MODEL_FOLDER_PATH, classifier_codename, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(self.MODEL_FOLDER_PATH, 'data', dataset_codename + '_label_map.pbtxt')

        # Max number of available category names in the model only for the COCO dataset
        self.COCO_MAX_CLASSES = 90
    
        # Placeholders:
        # -------------------------------------------------------------------- #
        # A dictionary of [IDs] => category names
        # i.e. If our CNN predicts `1`, we know that this corresponds to `person`.
        # - Check '../object_detection_api/data/mscoco_label_map.pbtxt' for a full list of category names
        self.categories_dict = None
        self.PEDESTRIAN = 1
        self.BIKES = [2, 4]
        self.VEHICLES = [3, 6, 7, 8, 9]
        self.TRAFFIC_LIGHT = 10
        self.STOP_SIGN = 13
        self.PARKING_METER = 14


        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = None

        # Each class refers to the ID of a particular category name for the object that was detected.
        self.detection_classes = None

        # Each score represent the level of confidence for each of the objects.
        # Note: scores will be shown on the result image, together with the class label.
        self.detection_scores = None

        # Number of objects detected within a given frame
        self.num_detections = None

        # The decision threshold : all detection scores below this given threshold will be discarded
        self.classifier_threshold = classifier_threshold

        self.frame = None
        # -------------------------------------------------------------------- #
        

    def download_model(self):
        """
        Download pre-trained model from the tensorflow object detection API.
        """
        if not os.path.exists(self.MODEL_FILE_PATH):
            print('\n\n-- Downloading classifier model...')
            opener = urllib.request.URLopener()
            opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE_PATH)


        if not os.path.exists(self.PATH_TO_CKPT):
            print('\n\n-- Extracting classifier model...')
            tar_file = tarfile.open(self.MODEL_FILE_PATH)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.path.join(self.MODEL_FOLDER_PATH))


    def load_model(self):
        """
        Load pre-trained model into memory.
        """
        print('\n\n-- Loading classifier model into memory...')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # label mapping - to map indices to category names,
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
            max_num_classes = self.COCO_MAX_CLASSES,
            use_display_name = True)
        self.categories_dict = label_map_util.create_category_index(categories)


    def setup(self):
        """
        This method would download a trained model from the API if necessary files were not found.
        And, it loads the trained model into memory - preferably GPU memory.
        Then, prep. the tensorflow computation graph and initiates a tensorflow session.
        """
        self.download_model()
        self.load_model()
        self.detection_graph.as_default()
        self.sess = tf.Session(graph = self.detection_graph)


    def threat_classifier(self):
        """
        Evaluate detected objects and return a dictionary to indicate any potential threats.
        """
        objects_dict = {
            "COLLISION": False,
            "PEDESTRIAN": False,
            "STOP_SIGN": False,
            "TRAFFIC_LIGHT": False,
            "VEHICLES": False,
            "BIKES": False,
        }

        frame_height, frame_width = self.frame.shape[:2]

        # get the detected objects by the neural network along with their confidence scores
        detected_objs = zip(self.detection_classes[0], self.detection_scores[0], self.detection_boxes[0])

        # region of interest
        # width = 1/4 of the frame's width starting from the center point and expanding 1/4 in each direction
        # height = 1/2 of the frame's height
        roi = {
            "t": frame_height/2,                    #top_boundary
            "l": (frame_width/2) - (frame_width/4), #left_boundary
            "b": frame_height,                      #bottom_boundary
            "r": (frame_width/2) + (frame_width/4), #right_boundary

            # COLLISION Detection Area
            # width = 1/4 of the frame's width starting from the center point and expanding 1/8 in each direction
            # height = 1/12 of the frame's bottom base
            "ct": frame_height - (frame_height/12),  #top_boundary
            "cl": (frame_width/2) - (frame_width/8), #left_boundary
            "cr": (frame_width/2) + (frame_width/8)  #right_boundary
        }
        
        if self.diagnostic_mode:
            # draw a box around the area scaned for for PEDESTRIAN/VEHICLES detection
            visualization_utils.draw_bounding_box_on_image_array(
                self.frame,
                roi["t"]/frame_height, roi["l"]/frame_width, roi["b"]/frame_height, roi["r"]/frame_width,
                color="Gold", display_str_list=(' ROI ',))

            # draw a box around the area scaned for collision warnings
            visualization_utils.draw_bounding_box_on_image_array(
                self.frame,
                roi["ct"]/frame_height, roi["cl"]/frame_width, roi["b"]/frame_height, roi["cr"]/frame_width,
                color="Red", display_str_list=(' COLLISION ROI ',))

        # update warning interface as needed 
        for(obj_id, confidence_score, pos) in detected_objs:
            if confidence_score >= self.classifier_threshold:
                # Locate object's 3D-spatial position according to its coordinates in the given frame
                obj_top = pos[0] * frame_height
                obj_left = pos[1] * frame_width
                obj_bottom = pos[2] * frame_height
                obj_right = pos[3] * frame_width

                object_width = obj_right - obj_left
                object_height = obj_bottom - obj_top

                # Collision
                # -------------------------------------------------------------------- #
                base_flag, cornor_flag, too_large_flag = False, False, False
                
                # if the bottom base of the object is below the upper boundary of the scanned area 
                if obj_bottom > roi["ct"]:
                    base_flag = True

                # if either corners of the object is within the scanned area
                if (obj_right < roi["cr"] and obj_right > roi["cl"]) or \
                    (obj_left < roi["cr"] and obj_left > roi["cl"]):
                    cornor_flag = True

                # if both corners of the object are around the scanned area
                if obj_left < roi["cl"] and obj_right > roi["cr"]:
                    too_large_flag = True

                if base_flag and (cornor_flag or too_large_flag):
                    objects_dict["COLLISION"] = True

                    if self.visualization:
                        # highlight object when there's a collision warning
                        matrix = np.zeros(self.frame.shape[:2])
                        for c in range(frame_width - 1):
                            for r in range(frame_height - 1):
                                if (c > obj_left and c < obj_right and r > obj_top and r < obj_bottom):
                                    matrix[r][c] = 1
                                    
                        mask = np.asarray(matrix, dtype=np.uint8)
                        visualization_utils.draw_mask_on_image_array(
                            self.frame, mask, color='Red', alpha=.5)

                        #visualization_utils.draw_bounding_box_on_image_array(
                        #    self.frame,
                        #    pos[0], pos[1], pos[2], pos[3],
                        #   thickness=50)

                # -------------------------------------------------------------------- #

                # classification
                # -------------------------------------------------------------------- #
                if obj_id == self.PEDESTRIAN:
                    # alert the driver if there's a pedestrian crossing in front of the car
                    if obj_bottom > roi["t"]:
                        objects_dict["PEDESTRIAN"] = True
                            
                elif obj_id == self.STOP_SIGN:
                    objects_dict["STOP_SIGN"] = True      

                elif obj_id == self.TRAFFIC_LIGHT:
                    objects_dict["TRAFFIC_LIGHT"] = True

                elif obj_id in self.VEHICLES:
                    if obj_bottom > roi["t"]:
                        objects_dict["VEHICLES"] = True

                elif obj_id in self.BIKES:
                    if obj_bottom > roi["t"]:
                        objects_dict["BIKES"] = True
                # -------------------------------------------------------------------- #
                
        return objects_dict


    def scan_road(self, frame, threats_dict):
        """
        Detect objects and classify them into one of the defined categories in the dataset.
        """
        self.frame = frame

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        frame_expanded = np.expand_dims(self.frame, axis=0)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Run session to get detections.
        (self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        
        # Run threat_classifier() method
        threats_dict.update(self.threat_classifier())

        if self.visualization:
            # Visualization of the results of a detection.
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                self.frame,
                np.squeeze(self.detection_boxes),
                np.squeeze(self.detection_classes).astype(np.int32),
                np.squeeze(self.detection_scores),
                self.categories_dict,
                use_normalized_coordinates=True,
                min_score_thresh=self.classifier_threshold, 
                line_thickness=1)        

        return (self.frame, threats_dict)
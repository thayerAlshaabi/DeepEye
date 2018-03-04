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
from object_classifier.lane_detection.LaneDetector import *
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

    - Window Management:
        We use a cross-platform API for captureing screenshots called (MSS)
        For more information about MSS-API refer to http://python-mss.readthedocs.io/examples.html

        Note: By default our calss will capture the entire main monitor and pass it into the CNN.
        - monitor_id: ID of the monitor to be captured
        - window_top_offset: Top Offset in pixels (0 by default)
        - window_left_offset: Left Offset in pixels (0 by default)
        - window_width: The desired width of the captured window (full width of the given monitor by default)
        - window_height: The desired height of the captured window (full height of the given monitor by default)
        - window_scale: A scaling factor for the captured window (1.0 by default)
    """
    # Constructor
    def __init__(self,
        classifier_codename = 'faster_rcnn_nas_coco_2017_11_08',
        dataset_codename = 'mscoco',
        classifier_threshold = .75,
        lane_detection = True,
        visualization = False,
        monitor_id = 1,
        window_top_offset = 0,
        window_left_offset = 0,
        window_width = None,
        window_height = None,
        window_scale = 1.0):

        # Boolean flags for visualization utils
        self.lane_detection = lane_detection
        self.visualization = visualization
        
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

        # Instance of the MSS-API for captureing screenshots
        self.window_manager = mss.mss()

        # Note:
        #   - monitor_id = 0 | grab all monitors together
        #   - monitor_id = n | grab a given monitor (n) : where n > 0
        self.target_window = self.window_manager.monitors[monitor_id]

        # Update position of the window that will be captured
        if window_left_offset:
            self.target_window['left'] += window_left_offset
            self.target_window['width'] -= window_left_offset
        if window_top_offset:
            self.target_window['top'] += window_top_offset
            self.target_window['height'] -= window_top_offset
        if window_width:
            self.target_window['width'] = window_width
        if window_height:
            self.target_window['height'] = window_height
        if window_scale:
            self.target_window['scale'] = window_scale


        # Placeholders:
        # -------------------------------------------------------------------- #
        # A dictionary of [IDs] => category names
        # i.e. If our CNN predicts `1`, we know that this corresponds to `person`.
        # - Check '../object_detection_api/data/mscoco_label_map.pbtxt' for a full list of category names
        self.categories_dict = None

        # Captured frame
        self.frame = None

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

        # Number of frames captured per second
        self.frame_rate = 0
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
        self.lane_detector = LaneDetector()
        print('\n\n-- Running object detector: target_window:', self.target_window)


    def scan_road(self):
        """
        Capture frames and detecte objects.
        """
        # Register current time to be used for calculating frame rate
        timer = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        pixels_arr = np.asarray(self.window_manager.grab(self.target_window))
        
        # convert pixels from BGRA to RGB values
        self.frame = cv2.cvtColor(pixels_arr, cv2.COLOR_BGRA2RGB)

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
        
        if self.visualization:
            # Visualization of the results of a detection.
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                self.frame,
                np.squeeze(self.detection_boxes),
                np.squeeze(self.detection_classes).astype(np.int32),
                np.squeeze(self.detection_scores),
                self.categories_dict,
                use_normalized_coordinates=True,
                min_score_thresh=self.classifier_threshold, line_thickness=1)

        # detect lane in the given frame
        if self.lane_detection:
            self.frame = self.lane_detector.detect_lane(self.frame)

        if self.visualization and not self.lane_detection:
            # convert to grayscale to reduce computational power needed for the process
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        # Display frame with detected objects.
        cv2.imshow('DeepEye | Obj-Detector', self.frame)

        # Calculating fps based on the previous registered timer
        self.frame_rate = 10 / (time.time() - timer)
        print('frame_rate: {0}'.format(int(self.frame_rate)))


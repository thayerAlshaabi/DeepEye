# coding: utf-8
"""
Our system consists of two major layers.
In the first layer, a deep convolutional neural network mainly used for object detection and image segmentation.
The network takes in a raw pixel image, then classify it into one of the labels defined in the dataset.
The output of the CNN will then be fed into the second layer,
which is going to perform online motion analysis and trajectory-based tracking
to detect any situations that may pose a potential threat to the driving agent.
Lastly, the result of our structure for motion (SfM) system will then be transformed
to voice warning system that would notify the driver of any upcoming threats.

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

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import cv2
import os, sys
import mss
import mss.tools
import time

from object_classifier.ObjectClassifier import *
from lane_detector.LaneDetector import *
from object_classifier.object_detection.utils import label_map_util, visualization_utils

# ---------------------------------------------------------------------------- #

class DrivingAssistant:
    # Constructor
    def __init__(self,
        classifier_codename,
        dataset_codename,
        classifier_threshold,
        object_detection = True,
        object_visualization = True,
        lane_detection = True,
        lane_visualization = True,
        diagnostic_mode = True,
        monitor_id = 1,
        window_top_offset = 0,
        window_left_offset = 0,
        window_width = None,
        window_height = None,
        window_scale = 1.0):

        # Boolean flag for feature-customization
        self.object_detection = object_detection
        self.object_visualization = object_visualization
        self.diagnostic_mode = diagnostic_mode

        self.lane_detection = lane_detection
        self.lane_visualization = lane_visualization

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

        print("Activating DeepEye Advanced Co-pilot Mode")
        
        self.object_detector = ObjectClassifier(
            classifier_codename = classifier_codename,
            dataset_codename = dataset_codename,
            classifier_threshold = classifier_threshold,
            visualization = object_visualization,
            diagnostic_mode = diagnostic_mode,
            frame_height = self.target_window['height'],
            frame_width = self.target_window['width']
        )

        self.lane_detector = LaneDetector(
            visualization = lane_visualization
        )

        self.threats = {
            "COLLISION": False,
            "PEDESTRIAN": False,
            "STOP_SIGN": False,
            "TRAFFIC_LIGHT": False,
            "VEHICLES": False,
            "BIKES": False,
            "FAR_LEFT": False,
            "FAR_RIGHT": False,
            "RIGHT": False,
            "LEFT": False,
            "CENTER": False,
            "UNKNOWN": True
        }

        self.frame_id = 0

        self.columns = [
            'FRAME_ID',
            'PEDESTRIAN',
            'VEHICLES',
            'BIKES',
            'STOP_SIGN',
            'TRAFFIC_LIGHT',
            'OFF_LANE',
            'COLLISION'
        ]

        self.data_frame = pd.DataFrame(columns=self.columns)


    def run(self):   
        """
        Capture frames, initiate both objects and lane detectors, and then visualize output. 
        """
        # Get raw pixels from the screen, save it to a Numpy array
        original_frame = np.asarray(self.window_manager.grab(self.target_window))

        # set frame's height & width
        frame_height, frame_width = original_frame.shape[:2]

        # convert pixels from BGRA to RGB values 
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)

        if self.diagnostic_mode:
            pre = original_frame.copy()
            # draw a box around the area scaned for for PEDESTRIAN/VEHICLES detection
            visualization_utils.draw_bounding_box_on_image_array(
                pre,
                self.object_detector.roi["t"]/frame_height, 
                self.object_detector.roi["l"]/frame_width, 
                self.object_detector.roi["b"]/frame_height, 
                self.object_detector.roi["r"]/frame_width,
                color=(255, 255, 0), # BGR VALUE 
                display_str_list=(' ROI ',))

            # draw a box around the area scaned for collision warnings
            visualization_utils.draw_bounding_box_on_image_array(
                pre,
                self.object_detector.roi["ct"]/frame_height, 
                self.object_detector.roi["cl"]/frame_width, 
                self.object_detector.roi["b"]/frame_height, 
                self.object_detector.roi["cr"]/frame_width,
                color=(255, 0, 255), # BGR VALUE 
                display_str_list=(' COLLISION ROI ',))

            # save a screen shot of the current frame before getting processed 
            if self.frame_id % 10 == 0:
                cv2.imwrite("test/pre/" + str(self.frame_id/10) + ".jpg", pre)

        # only detect objects in the given frame
        if self.object_detection and not self.lane_detection:
            (frame, self.threats) = self.object_detector.scan_road(original_frame, self.threats)

            if self.object_visualization:
                # Display frame with detected objects.
                cv2.imshow(
                    'DeepEye Dashboard', 
                    cv2.resize(frame, (640, 480))
                )

        # only detect lane in the given frame
        elif self.lane_detection and not self.object_detection:
            (frame, self.threats) = self.lane_detector.detect_lane(original_frame, self.threats)

            if self.lane_visualization:
                # Display frame with detected lane.
                cv2.imshow(
                    'DeepEye Dashboard', 
                    cv2.resize(frame, (640, 480))
                )
        
        # detect both objects and lane
        elif self.object_detection and self.lane_detection:

            # Visualize object detection ONLY 
            if self.object_visualization and not self.lane_visualization:
                (frame, self.threats) = self.object_detector.scan_road(original_frame, self.threats)

                # Display frame with detected lane.
                cv2.imshow(
                    'DeepEye Dashboard', 
                    cv2.resize(frame, (640, 480))
                )

                (_, self.threats) = self.lane_detector.detect_lane(original_frame, self.threats)
                
                
            # Visualize lane detection ONLY 
            elif self.lane_visualization and not self.object_visualization:
                (frame, self.threats) = self.lane_detector.detect_lane(original_frame, self.threats)

                # Display frame with detected lane.
                cv2.imshow(
                    'DeepEye Dashboard', 
                    cv2.resize(frame, (640, 480))
                )

                (_, self.threats) = self.object_detector.scan_road(original_frame, self.threats)

            # Visualize both object & lane detection 
            elif self.object_visualization and self.lane_visualization:      
                (frame, self.threats) = self.object_detector.scan_road(original_frame, self.threats)
                (frame, self.threats) = self.lane_detector.detect_lane(frame, self.threats)

                # Display frame with detected lane.
                cv2.imshow(
                    'DeepEye Dashboard', 
                    cv2.resize(frame, (640, 480))
                )

            # skip visualization
            else:
                (_, self.threats) = self.object_detector.scan_road(original_frame, self.threats)
                (_, self.threats) = self.lane_detector.detect_lane(original_frame, self.threats)
        
        # skip detection
        else:
            frame = original_frame

        
        if (self.frame_id % 10 == 0) and (self.diagnostic_mode):
            # draw a box around the area scaned for for PEDESTRIAN/VEHICLES detection
            visualization_utils.draw_bounding_box_on_image_array(
                frame,
                self.object_detector.roi["t"]/frame_height, 
                self.object_detector.roi["l"]/frame_width, 
                self.object_detector.roi["b"]/frame_height, 
                self.object_detector.roi["r"]/frame_width,
                color=(255, 255, 0), # BGR VALUE 
                display_str_list=(' ROI ',))

            # draw a box around the area scaned for collision warnings
            visualization_utils.draw_bounding_box_on_image_array(
                frame,
                self.object_detector.roi["ct"]/frame_height, 
                self.object_detector.roi["cl"]/frame_width, 
                self.object_detector.roi["b"]/frame_height, 
                self.object_detector.roi["cr"]/frame_width,
                color=(255, 0, 255), # BGR VALUE 
                display_str_list=(' COLLISION ROI ',))

            # save a screen shot of the current frame after getting processed 
            cv2.imwrite("test/post/" + str(self.frame_id/10) + ".jpg", frame)
            
            if self.threats["FAR_LEFT"] or self.threats["FAR_RIGHT"]:
                OFF_LANE = 1
            else:
                OFF_LANE = 0

            # append a new row to dataframe
            self.data_frame = self.data_frame.append({
                'FRAME_ID':         int(self.frame_id/10),
                'PEDESTRIAN':       int(self.threats['PEDESTRIAN']),
                'VEHICLES':         int(self.threats['VEHICLES']),
                'BIKES':            int(self.threats['BIKES']),
                'STOP_SIGN':        int(self.threats['STOP_SIGN']),
                'TRAFFIC_LIGHT':    int(self.threats['TRAFFIC_LIGHT']),
                'OFF_LANE':         int(OFF_LANE),
                'COLLISION':        int(self.threats['COLLISION']),
            }, ignore_index=True)

        self.frame_id += 1

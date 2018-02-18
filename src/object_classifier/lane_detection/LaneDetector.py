# coding: utf-8
"""
This class uses a lot of OpenCV built-in functions 
to detect the current lane that the car is driving in, 
then highlights both the road markers, as well as, 
the area enclosed by your lane onto the given frame.

----------------------

# Licensing Information:
The following code was adapted from the Udacity CarND Project

# Original SourceCode:
https://github.com/udacity/CarND-Advanced-Lane-Lines

"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import cv2
import numpy as np

# imports from the lane detection module.
import object_classifier.lane_detection.calibration_utils as calibrator
import object_classifier.lane_detection.graphic_utils as transformer
import object_classifier.lane_detection.visualization_utils as visualizer
# ---------------------------------------------------------------------------- #

class LaneDetector:
    # Constructor
    def __init__(self):
        self.lane_visualizer = visualizer.Lane()

        # setup calibration parameters
        # -------------------------------------------------------------------- #
        self.ret,\
        self.camera_matrix,\
        self.distortion_coefficients,\
        self.rotation_vectors,\
        self.translation_vectors = calibrator.get_prams()
        # -------------------------------------------------------------------- #


    def detect_lane(self, frame):
        """
        Mark the area enclosed by your lane onto the given frame.
        """
        height, width = frame.shape[:2]

        # adjust calibration prams to the given frame 
        adjusted_frame = calibrator.set_distortion_coefficients(
            frame,
            self.camera_matrix, 
            self.distortion_coefficients)

        # highlight lanes in the frame
        lanes_bitmap = transformer.convert_to_bitmap(adjusted_frame)

        # compute transformation matrices to get bird's eye view
        birdseye_view, forward_transformation_matrix, backward_transformation_matrix = transformer.convert_to_birdseye_view(lanes_bitmap)

        #run a sliding window search to detect lane in the frame  
        lane_was_detected = self.lane_visualizer.detect_pixles(birdseye_view)

        # highlight lane onto the given frame if it was detected
        if lane_was_detected:
            output = self.lane_visualizer.highlight(
                adjusted_frame, 
                backward_transformation_matrix)

            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        else: # otherwise return the original frame 
            return cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)
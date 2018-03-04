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
        self.lane = visualizer.Lane()

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

        # run a sliding window search to detect lane in the frame  
        self.lane.detect_pixles(birdseye_view)
        
        # evaluate the current situation for any potential threats
        potential_threat_level = self.threat_classifier(adjusted_frame)

        # highlight lane onto the given frame if it was detected
        # if car is off-lane => highlight lane in red to alert the driver
        if potential_threat_level == 2 or potential_threat_level == -2:  
            print("-- Lane-Departure Warning")
            output = self.lane.highlight(
                adjusted_frame, 
                backward_transformation_matrix,
                lane_color=(255, 0, 0))
            
        # if car is slightly off-lane => highlight lane in orange
        elif potential_threat_level == 1 or potential_threat_level == -1:
            print("-- Off-Lane Warning")
            output = self.lane.highlight(
                adjusted_frame, 
                backward_transformation_matrix,
                lane_color=(255, 127, 0))

        else: # if car is relatively in the center of lane => highlight lane in green
            output = self.lane.highlight(
                adjusted_frame, 
                backward_transformation_matrix,
                lane_color=(0, 255, 127))

        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    

    def threat_classifier(self, frame):
        """
        Evaluate the current situation for any potential threats
            - return -2 :: if car is off-lane (left-side)
            - return -1 :: if car is slightly off-lane (left-side)
            - return  0 :: if car is relatively in the center of lane
            - return  1 :: if car is slightly off-lane (right-side)
            - return  2 :: if car is off-lane (right-side)
        """
        threat_level = 0 
        current_pos = 0
        monitor_ratio = frame.shape[0]/frame.shape[1]

        if self.lane.lane_detected:

            # calculate the right and left boundaries of the given lane 
            left_boundary = np.mean( 
                self.lane.left_marker.x_axis_pixels
                [
                    self.lane.left_marker.y_axis_pixels > 0.95 \
                    * self.lane.left_marker.y_axis_pixels.max()
                ]
            )
            
            right_boundary = np.mean(
                self.lane.right_marker.x_axis_pixels
                [
                    self.lane.right_marker.y_axis_pixels > 0.95 \
                    * self.lane.right_marker.y_axis_pixels.max()
                ]
            )
            
            # calculate the width of the lane
            width = right_boundary - left_boundary
            
            center_point = frame.shape[1] / 2   

            # calculate the offset from the center point of the given lane
            current_pos = ((left_boundary + width / 2) - center_point) * monitor_ratio
        

        # if car is off-lane (right-side)
        if current_pos >= 75:
            threat_level = 2
                    
        # if car is off-lane (left-side)
        elif current_pos <= -75:
            threat_level =  -2
           
        # if car is slightly off-lane (right-side)
        elif current_pos > 50 and current_pos < 75:
            threat_level =  1
            
        # if car is slightly off-lane (left-side)
        elif current_pos < -50 and current_pos > -75:
            threat_level =  -1
            
        # if car is relatively in the center of lane
        else:
            threat_level = 0

        return threat_level
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
import lane_detector.calibration_utils as calibrator
import lane_detector.graphic_utils as transformer
import lane_detector.visualization_utils as visualizer
# ---------------------------------------------------------------------------- #

class LaneDetector:
    # Constructor
    def __init__(self,
        visualization = True):

        self.lane = visualizer.Lane()

        # Boolean flag for visualization utils
        self.visualization = visualization

        # setup calibration parameters
        # -------------------------------------------------------------------- #
        self.ret,\
        self.camera_matrix,\
        self.distortion_coefficients,\
        self.rotation_vectors,\
        self.translation_vectors = calibrator.get_prams()
        # -------------------------------------------------------------------- #

        self.frame = None


    def threat_classifier(self):
        """
        Evaluate the current situation for any potential threats
        return a dict {
            "FAR_LEFT": if car is off-lane (left-side)
            "FAR_RIGHT": if car is off-lane (right-side)
            "RIGHT": if car is slightly off-lane (right-side)
            "LEFT": if car is slightly off-lane (left-side)
            "CENTER": if car is relatively in the center of lane
            "UNKNOWN": if lane was not detected
        }
        """
        lane_dict = {
                "FAR_LEFT": False,
                "FAR_RIGHT": False,
                "RIGHT": False,
                "LEFT": False,
                "CENTER": False,
                "UNKNOWN": False
        }

        current_pos = 0
        monitor_ratio = self.frame.shape[0]/self.frame.shape[1]

        if self.lane.lane_detected:
            # calculate the right and left boundaries of the given lane 
            if len(self.lane.left_marker.x_axis_pixels) > 0 and \
                len(self.lane.left_marker.y_axis_pixels) > 0 and \
                len(self.lane.right_marker.x_axis_pixels) > 0 and \
                len(self.lane.right_marker.y_axis_pixels) > 0:

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
                # false detection
                if width < 400 or width > 1100:
                    lane_dict["UNKNOWN"] = True
                    
                else:
                    center_point = self.frame.shape[1] / 2   
                    # calculate the offset from the center point of the given lane
                    current_pos = ((left_boundary + width / 2) - center_point) * monitor_ratio
                    lane_dict["UNKNOWN"] = False

            else:
                lane_dict["UNKNOWN"] = True

        else:
            lane_dict["UNKNOWN"] = True


        # if car is off-lane (right-side)
        if current_pos >= 75:
            lane_dict["FAR_RIGHT"] = True
                    
        # if car is off-lane (left-side)
        elif current_pos <= -75:
            lane_dict["FAR_LEFT"] = True
            
        # if car is slightly off-lane (right-side)
        elif current_pos > 50 and current_pos < 75:
            lane_dict["RIGHT"] = True
            
        # if car is slightly off-lane (left-side)
        elif current_pos < -50 and current_pos > -75:
            lane_dict["LEFT"] = True
            
        # if car is relatively in the center of lane
        else:
            lane_dict["CENTER"] = True

        return lane_dict


    def detect_lane(self, frame, threats_dict):
        """
        Mark the area enclosed by your lane onto the given frame.
        """
        self.frame = frame

        height, width = self.frame.shape[:2]

        # adjust calibration prams to the given frame 
        self.calb_frame = calibrator.set_distortion_coefficients(
            self.frame,
            self.camera_matrix, 
            self.distortion_coefficients)

        # highlight lanes in the frame
        lanes_bitmap = transformer.convert_to_bitmap(self.calb_frame)

        # compute transformation matrices to get bird's eye view
        birdseye_view, forward_transformation_matrix, backward_transformation_matrix = transformer.convert_to_birdseye_view(lanes_bitmap)

        # run a sliding window search to detect lane in the frame  
        self.lane.detect_pixles(birdseye_view)
        
        # evaluate the current situation for any potential threats
        threats_dict.update(self.threat_classifier())

        # highlight lane onto the given frame if it was detected
        if self.visualization:
            # lane was not detected
            if threats_dict["UNKNOWN"]: 
                pass

            # if car is off-lane => highlight lane in red to alert the driver
            elif threats_dict["FAR_RIGHT"] or threats_dict["FAR_LEFT"]:  
                self.frame = self.lane.highlight(
                    self.frame, 
                    backward_transformation_matrix,
                    lane_color=(255, 0, 0))
                
            # if car is slightly off-lane => highlight lane in orange
            elif threats_dict["RIGHT"] or threats_dict["LEFT"]:  
                self.frame = self.lane.highlight(
                    self.frame, 
                    backward_transformation_matrix,
                    lane_color=(255, 127, 0))

            # if car is relatively in the center of lane => highlight lane in green
            elif threats_dict["CENTER"]: 
                self.frame = self.lane.highlight(
                    self.frame, 
                    backward_transformation_matrix,
                    lane_color=(0, 255, 127))
        else:
            pass

        return (cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), threats_dict)
    

    
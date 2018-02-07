# coding: utf-8
"""
The functions below use OpenCV functions to mark/highlight 
the area enclosed by your lane onto the given frame.

Sources:
- cv2.fillPoly: fills an area bounded by several polygonal contours.
    https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html#fillpoly

- cv2.getPerspectiveTransform & cv.warpPerspective: convert frame to a wrapped/flattened bird's eye view
    https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html

- cv2.addWeighted: update specific pixels within a frame
    https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype)
"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import numpy as np
import cv2
import glob
import collections
# ---------------------------------------------------------------------------- #

class RoadMarker:
    """
    A generic class for to define lane dividers. 
    """
    def __init__(self, size, color):
        # size of the lane marker
        self.size = size

        # color of lane marker 
        self.color = color

        # pixel and polynomial coefficients of the last iteration
        self.last_observed_pixel = None
        self.last_polygons_coefficients = None

        # a list of the last 10 pixels and their corresponding polynomial coefficients
        self.recent_pixels = collections.deque(maxlen=10)
        self.recent_polygons_coefficients = collections.deque(maxlen=20)
        
        # a list of all observed_pixels
        self.x_axis_pixels = None
        self.y_axis_pixels = None


    def mark(self, frame):
        """
        Draw a line on the given frame. 
        """
        # return an array of evenly spaced numbers over a specified interval (height of the frame).
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
        metronome = np.linspace(0, frame.shape[0]-1, frame.shape[0])

        center = self.last_observed_pixel[0] * metronome ** 2 \
            + self.last_observed_pixel[1] * metronome \
            + self.last_observed_pixel[2]

        left_boundary = center - self.size // 2
        right_boundary = center + self.size // 2

        # reformat arrays to use (cv2.fillPoly)
        left_points = np.array(list(zip(left_boundary, metronome)))
        right_points = np.array(np.flipud(list(zip(right_boundary, metronome))))

        # array of polygons where each polygon is represented as an array of points.
        polygons_points = [np.int32(np.vstack([left_points, right_points]))]

        return cv2.fillPoly(frame, polygons_points, self.color)

    
    def adjust(self, new_px, new_polygons_coefficients, reset=False):
        """
        Update line boundaries of the lane as needed. 
        """
        # empty out the arrays of last observations 
        if reset:
            self.recent_pixels = []
            self.recent_polygons_coefficients = []

        # re-assign member variables 
        self.last_observed_pixel = new_px
        self.last_polygons_coefficients = new_polygons_coefficients
        
        self.recent_pixels.append(self.last_observed_pixel)
        self.recent_polygons_coefficients.append(self.last_polygons_coefficients)


class Lane:
    """
    A wrapper class to maintain all information about the lane 
    
    -----
    Note: The constructor doesn't necessarily require passing any parameters as
    they all have some satisfactory default values to start with.
    """
    def __init__(self,
        marker_size = 50, 
        marker_color = (255, 255, 255),
        lane_color = (0, 255, 127),
        x_polygons_coefficients = 3.7/700, 
        y_polygons_coefficients = 30/720,
        num_windows = 9):
        
        # width of the lane marker
        self.marker_size = marker_size

        # color of lane marker 
        self.marker_color = marker_color

        # color of the area enclosed yb your lane
        self.lane_color = lane_color

        # coordinates of the region of interest
        self.x_polygons_coefficients = x_polygons_coefficients
        self.y_polygons_coefficients = y_polygons_coefficients

        # a boolean flag to indicate whether the lane was detected during the last iteration or not
        self.lane_detected = False

        # define two lane dividers 
        self.left_marker = RoadMarker(self.marker_size, self.marker_color) 
        self.right_marker = RoadMarker(self.marker_size, self.marker_color) 

        self.num_windows = num_windows
        
    
    def detect_pixles(self, birdeye_view):
        """
        Get the target pixels that encapsulate both road markers in the given frame. 
        """
        frame_height, frame_width = birdeye_view.shape

        # set the width & height of the windows used for search
        self.window_height = np.int(frame_height / self.num_windows)
        self.window_width = 100

        # minimum number of pixels found to recenter window
        self.min_window_pixels = 50

        # create a histogram of the bottom half of the frame
        pixels_histogram = np.sum(birdeye_view[frame_height//2:-30, :], axis=0)

        # locate the mid-point of the histogram
        mid_point = len(pixels_histogram) // 2

        # locate the peak point of the left-half of the histogram, 
        # which will be the base point of the left lane divider 
        left_pos = np.argmax(pixels_histogram[:mid_point])

        # locate the peak point of the right-half of the histogram, 
        # which will be the base point of the right lane divider 
        right_pos = np.argmax(pixels_histogram[mid_point:]) + mid_point

        # create two empty lists to store the indices of some candidate pixels
        # in order to retrieve those pixels when needed 
        left_candidate_pixels = []
        right_candidate_pixels = []

        # keep track of the slider position for each window
        slider_position = (left_pos, right_pos)

        # collect all non-zero pixels and store their coordinates
        target_pixels = birdeye_view.nonzero()
        y_target_pixels = np.array(target_pixels[0])
        x_target_pixels = np.array(target_pixels[1])

        # iterate through each window
        for window in range(self.num_windows):
            # define the current window position onto the frame 
            # -------------------------------------------------------------------- #
            # x: Bottom - Left
            xbl = slider_position[0] - self.window_width
            # x: Top - Left
            xtl = slider_position[0] + self.window_width
            # x: Bottom - Right
            xbr = slider_position[1] - self.window_width
            # x: Top - Right
            xtr = slider_position[1] + self.window_width
            # y: Bottom
            yb = frame_height - (window + 1) * self.window_height
            # y: Top
            yt = frame_height - window * self.window_height
            # -------------------------------------------------------------------- #

            # collect non-zero pixels located in the current target window
            left_candidate_pixels.append(
                ((y_target_pixels >= yb) 
                & (y_target_pixels < yt) 
                & (x_target_pixels >= xbl)
                & (x_target_pixels < xtl)).nonzero()[0]
            )

            right_candidate_pixels.append(
                ((y_target_pixels >= yb) 
                & (y_target_pixels < yt) 
                & (x_target_pixels >= xbr)
                & (x_target_pixels < xtr)).nonzero()[0]
            )

            # if the window has a lot of 'hot' pixels 
            # take the mean of their position
            # and reset the silder's position to that center point
            if len(left_candidate_pixels) > self.min_window_pixels:
                slider_position[0] = np.int(np.mean(x_target_pixels[left_candidate_pixels]))

            if len(right_candidate_pixels) > self.min_window_pixels:
                slider_position[1] = np.int(np.mean(x_target_pixels[right_candidate_pixels]))

         # join/concatenate candidate_pixels arrays along the x-axis.
        left_candidate_pixels = np.concatenate(left_candidate_pixels)
        right_candidate_pixels = np.concatenate(right_candidate_pixels)

        # get road marker pixel positions
        self.left_marker.x_axis_pixels = x_target_pixels[left_candidate_pixels]
        self.left_marker.y_axis_pixels = y_target_pixels[left_candidate_pixels]

        self.right_marker.x_axis_pixels = x_target_pixels[right_candidate_pixels]
        self.right_marker.y_axis_pixels = y_target_pixels[right_candidate_pixels]

        # reset flag
        self.lane_detected = True

        # if the left road marker was not detected in the last iteration 
        if not list(self.left_marker.x_axis_pixels) or \
            not list(self.left_marker.y_axis_pixels):

            # reset variables 
            new_px_left = self.left_marker.last_observed_pixel
            new_coefficients_left = self.left_marker.last_polygons_coefficients
            self.lane_detected = False
        
        else: # otherwise, apply least squares polynomial fit
            # returns a vector of coefficients that minimizes the squared error.
            new_px_left = np.polyfit(self.left_marker.y_axis_pixels, 
                self.left_marker.x_axis_pixels, 2)

            new_coefficients_left = np.polyfit(self.left_marker.y_axis_pixels * self.y_polygons_coefficients, 
                self.left_marker.x_axis_pixels * self.x_polygons_coefficients, 2)

        # if the right road marker was not detected in the last iteration
        if not list(self.right_marker.x_axis_pixels) or \
            not list(self.right_marker.y_axis_pixels):
            
            new_px_right = self.right_marker.last_observed_pixel
            new_coefficients_right = self.right_marker.last_polygons_coefficients
            self.lane_detected = False

        else: # otherwise, apply least squares polynomial fit
            # returns a vector of coefficients that minimizes the squared error.
            new_px_right = np.polyfit(self.right_marker.y_axis_pixels, 
                self.right_marker.x_axis_pixels, 2)

            new_coefficients_right = np.polyfit(self.right_marker.y_axis_pixels * self.y_polygons_coefficients, 
                self.right_marker.x_axis_pixels * self.x_polygons_coefficients, 2)

        # update road marker onto the frame as needed
        self.left_marker.adjust(new_px_left, new_coefficients_left, self.lane_detected)
        self.right_marker.adjust(new_px_right, new_coefficients_right, self.lane_detected)

        return self.lane_detected


    def highlight(self, undistorted_frame, backward_transformation_matrix):
        """
        Mark the area enclosed by your lane onto the given frame.
        """
        frame_height, frame_width, _ = undistorted_frame.shape

        # Generate x and y values for plotting
        metronome = np.linspace(0, frame_height - 1, frame_height)

        left = self.left_marker.last_observed_pixel[0] * metronome ** 2 + \
            self.left_marker.last_observed_pixel[1] * metronome + \
            self.left_marker.last_observed_pixel[2]

        right = self.right_marker.last_observed_pixel[0] * metronome ** 2 + \
            self.right_marker.last_observed_pixel[1] * metronome + \
            self.right_marker.last_observed_pixel[2]
        
        warpped_frame = np.zeros_like(undistorted_frame, dtype=np.uint8)

        # reformat arrays to use (cv2.fillPoly)
        left_points = np.array([np.transpose(np.vstack([left, metronome]))])
        right_points = np.array([np.flipud(np.transpose(np.vstack([right, metronome])))])

        # array of polygons where each polygon is represented as an array of points.
        polygons_points = np.hstack((left_points, right_points))

        # mark the area enclosed by your lane onto the given frame
        cv2.fillPoly(warpped_frame, np.int_([polygons_points]), self.lane_color)

        # unwarp frame to get the original image(frame)
        lane_mask = cv2.warpPerspective(warpped_frame, 
            backward_transformation_matrix, 
            (frame_width, frame_height))  

        # apply lane mask to the frame 
        output_frame = cv2.addWeighted(
            src1 = undistorted_frame, 
            alpha = 1.0, 
            src2 = lane_mask, 
            beta = 0.3, 
            gamma = 0.0)

        # mark the each road divider for the given lane
        warpped_frame = self.left_marker.mark(warpped_frame)
        warpped_frame = self.right_marker.mark(warpped_frame)
        
        # unwarp frame to get the original image(frame)
        normal_frame = cv2.warpPerspective(warpped_frame, 
            backward_transformation_matrix, 
            (frame_width, frame_height))  

        # copy the last updated frame  
        road_markers_mask = output_frame.copy()
        # get target pixels that need to get updated 
        target_pixels = np.any([normal_frame != 0][0], axis=2) 
        road_markers_mask[target_pixels] = normal_frame[target_pixels]

        # apply road markers mask to the frame 
        output_frame = cv2.addWeighted(
            src1 = road_markers_mask, 
            alpha = 0.8, 
            src2 = output_frame, 
            beta = 0.5, 
            gamma = 0.0)

        return output_frame



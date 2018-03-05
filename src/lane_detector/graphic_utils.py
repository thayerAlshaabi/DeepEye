# coding: utf-8
"""
The functions below use OpenCV functions to detect edges in a given frame.

Sources:
- cv2.equalizeHist: improve the contrast of the frame
    https://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html

- cv2.Canny: edge detection.
    https://docs.opencv.org/3.3.1/da/d22/tutorial_py_canny.html

- cv2.GaussianBlur: removing gaussian noise from the frame.
    https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

- cv2.getPerspectiveTransform & cv.warpPerspective: convert frame to a wrapped/flattened bird's eye view
    https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html

"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import cv2
import numpy as np
import glob
# ---------------------------------------------------------------------------- #

def get_yellow_lanes(frame):
    """
    Apply HSV mask to a frame to highlight yellow lanes in the frame.

    OpenCV Docs:
        https://docs.opencv.org/3.0.0/df/d9d/tutorial_py_colorspaces.html
    """
    min_value = np.array([0, 70, 70]) # light red
    max_value = np.array([50, 255, 255]) # light blue

    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of yellow color in hsv
    lower_threshold = np.all(hsv > min_value, axis=2)
    upper_threshold = np.all(hsv < max_value, axis=2)

    # define a threshold to get only yellow colors.
    mask = np.logical_and(lower_threshold, upper_threshold)

    return mask


def get_white_lanes(frame):
    """
    Apply histogram equalization to a frame to improve the contrast of the frame
    then threshold it, which will help to highlight the white lines in the frame

    OpenCv Docs:
        https://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html
    """
    # convert BGR to GrayScale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    equalization_histogram = cv2.equalizeHist(grayscale)

    _, mask = cv2.threshold(equalization_histogram, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    return mask


def canny_edge_detection(frame, sigma=0.33, kernel_size=3):
    """
    Apply Canny edge detection to a frame, then threshold the result
    to highlight only the edges points in the frame

    OpenCV Docs:
        Canny - https://docs.opencv.org/3.3.1/da/d22/tutorial_py_canny.html
        GaussianBlur - https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    """
    # convert BGR to GrayScale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blurring is highly effective in removing gaussian noise from the frame.
    # kernel_size: should be positive and odd
    blurred = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # compute the median of the pixles intensity
    median = np.median(blurred)

    # calculate thresholds using the computed median
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))

    #apply Canny edge detection
    return cv2.Canny(blurred, lower_threshold, upper_threshold)


def convert_to_bitmap(frame, kernel_size=3):
    """
    Convert a input frame to a bitmap (2D-array of zeros and ones) 
    where only the pixels corresponding to lanes in the frame 
    would be highlighted by assigning them with (ones)
    and the rest of the pixels would be ignored by assigning them to (zeros).
    """
    # create an empty bitmap and initialize all its values to zeros
    lanes_bitmap = np.zeros(shape=frame.shape[:2], dtype=np.uint8)

    # highlight yellow lanes by threshold in HSV color space
    lanes_bitmap = np.logical_or(lanes_bitmap, get_yellow_lanes(frame))

    # highlight white lanes by thresholding the equalized frame
    lanes_bitmap = np.logical_or(lanes_bitmap, get_white_lanes(frame))

    # apply canny_edge_detection algorithm to enhance lanes detection 
    lanes_bitmap = np.logical_or(lanes_bitmap, canny_edge_detection(frame, kernel_size))

    # apply a morphological transformation to paint/fill the small gaps in the detected lines
    # basically to get a (solid line) instead of an (intermittent or dashed line)
    # for more information please check out this page:
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html 
    bitmap = cv2.morphologyEx(
        lanes_bitmap.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        np.ones((kernel_size, kernel_size), np.uint8))

    return bitmap


def convert_to_birdseye_view(frame):
    """
    Convert frame to a wrapped/flattened bird's eye view.
    """
    height, width = frame.shape[:2]

    # define 4 points in the original space 
    src = np.float32([[width, height],   
                      [0, height],        
                      [width/3, height * (2/3)], 
                      [width/2, height * (2/3)]]) 

    # define 4 points in the warped space 
    dst = np.float32([[width, height],     
                      [0, height],     
                      [0, 0],       
                      [width, 0]])    


    # compute transformation matrices
    forward_transformation_matrix = cv2.getPerspectiveTransform(src, dst)
    backward_transformation_matrix = cv2.getPerspectiveTransform(dst, src)

    # warp frame to get bird's eye view 
    birdseye_view = cv2.warpPerspective(frame,
        forward_transformation_matrix, 
        (width, height), 
        flags=cv2.INTER_LINEAR)

    return birdseye_view, forward_transformation_matrix, backward_transformation_matrix

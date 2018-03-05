# coding: utf-8
"""
The functions below use OpenCV functions to do the following:
    - Project 3D points to the image plane given intrinsic and extrinsic parameters.
    
    - Compute extrinsic parameters given intrinsic parameters, 
        a few 3D points, and their projections.

    - Estimate intrinsic and extrinsic camera parameters 
        from several views of a known calibration pattern 
        (every view is described by several 3D-2D point correspondences).

    - Estimate the relative position and orientation of the stereo camera "heads" 
        and compute the rectification* transformation that makes the camera optical axes parallel.

----------------------

Sources:
- cv2.findChessboardCorners
    https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a

- cv2.calibrateCamera
    https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html#ga687a1ab946686f0d85ae0363b5af1d7b

- cv2.undistort
    https://docs.opencv.org/3.3.1/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d

----------------------

# Licensing Information:
For further information about camera calibration, which is also known as pinhole camera model, 
please check out OpenCV documentation page: 

Camera Calibration and 3D Reconstruction (https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html)
"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import cv2
import numpy as np
import glob
import pickle
import os, os.path as path
# ---------------------------------------------------------------------------- #

#constant parameters
# ---------------------------------------------------------------------------- #
# relative path to chessboard images that are used for calibration 
CALIBRATION_MODULE_PATH = path.join(
    os.getcwd(), 
    'lane_detector', 
    'camera_cal')

# path for a pickle data file to avoid recalibrating every time.
CACHED_DATA_PATH = path.join(CALIBRATION_MODULE_PATH, 'calibration_data.pickle')

# a list of chessboard images
chessboard_images = glob.glob(path.join(CALIBRATION_MODULE_PATH, 'calibration*.jpg'))

CHESSBOARD_WIDTH = 9
CHESSBOARD_HEIGHT = 6
CHESSBOARD_SIZE = (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT)
# ---------------------------------------------------------------------------- #


def calibration_decorator(calibration_function):
    """
    A decorator for calibration function to avoid re-computing calibration every time.
    """     
    def wrapper(*args, **kwargs):
        # check if pickle data file was previously computed 
        if path.exists(CACHED_DATA_PATH):
            print('\n\n-- Loading cached calibration data into memory...')
            # load cached data into memory 
            with open(CACHED_DATA_PATH, 'rb') as data:
                calibration = pickle.load(data)
        else:
            print('\n\n-- Recalibrating camera...')
            calibration = calibration_function(*args, **kwargs)
            # load cached data into memory 
            with open(CACHED_DATA_PATH, 'wb') as data:
                pickle.dump(calibration, data)

        return calibration

    return wrapper


@calibration_decorator
def get_prams():
    """
    - camera_matrix => 3x3 floating-point camera matrix 
    
    - distortion_coefficients => an array of distortion coefficients

    - rotation_vectors =>
        an array of rotation vectors estimated for each pattern view
        each k-th rotation vector together with the corresponding k-th translation vector
        brings the calibration pattern from the model coordinate space (in which object_3d_projection are specified) 
        to the world coordinate space, that is, a real position of the calibration pattern 
        in the k-th pattern view (k=0.. M -1).

    - translation_vectors => an array of translation vectors estimated for each pattern view.
    """
    # a 3D array of (x,y,z) coordinates projection of the object in real world space
    object_points = []  

    # a 2D array of (x,y) coordinates of the object on a 2D plane
    image_points = []

    # initialize a 3d array with zeros as a placeholder
    projection_points = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)

    # populate the array with indexes as follows:
    # [(0,0,0), (1,0,0) .... (0,1,0), (1,1,0) .... (0,2,0), (1,2,0) .... (8,5,0)]
    projection_points[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2)

    # Step through the list and search for chessboard corners
    for image in chessboard_images:

        original_image = cv2.imread(image)

        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # locate corners into the chessboard image
        pattern_was_found, detected_corners = cv2.findChessboardCorners(
            grayscale_image, 
            CHESSBOARD_SIZE, 
            None)

        # append corresponding coordinates if patterns was found
        if pattern_was_found:
            object_points.append(projection_points)
            image_points.append(detected_corners)

    # calibrating camera useing OpenCV cv2.calibrateCamera(...)
    ret,\
    camera_matrix,\
    distortion_coefficients,\
    rotation_vectors,\
    translation_vectors = cv2.calibrateCamera(
        object_points, 
        image_points, 
        grayscale_image.shape[::-1], 
        None, None)

    # return calibration parameters
    return ret,\
        camera_matrix,\
        distortion_coefficients,\
        rotation_vectors,\
        translation_vectors


def set_distortion_coefficients(frame, camera_matrix, distortion_coefficients):
    """
    The function transforms an image to compensate radial and tangential lens distortion.
    """
    return cv2.undistort(
        frame, 
        camera_matrix, 
        distortion_coefficients)

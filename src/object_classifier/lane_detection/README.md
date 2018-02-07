# DeepEye: LaneDetector

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
3. [Methods](#methods)
4. [Customization](#customization)
5. [Licensing Information](#licensing-information)


## Introduction
This class uses a lot of OpenCV built-in functions to detect the current lane that the car is driving in, then highlights both the road markers, as well as, the area enclosed by your lane onto the given frame.


## Algorithms


## Methods
Name | Description 
--- | ---
**\_\_init\_\_** | The constructor doesn't necessarily require passing any parameters as they all have some satisfactory default values to start with. See [Customization](#customization) for detailed information. 
**detect_lane** | Detect the current lane that the car is driving in.


## Customization
To use a different classifier and/or different dataset you'll need to look at:

Parameter | Description 
--- | ---
**marker_color** | A Tuple of RGB values to indicate the color used to mark the lane divider. <br/> **(255, 255, 255)[White]** by default.
**lane_color** | A Tuple of RGB values to indicate the color used to mark the area enclosed by your lane. <br/> **(0, 255, 127)[Light Green]** by default.


## Licensing Information
The following code was inspired by [Udacity: self-driving-car project](https://github.com/udacity/CarND-Advanced-Lane-Lines).

Original SourceCode was adopted from @ [advanced_lane_finding](https://github.com/ndrplz/self-driving-car) by Andrea Palazzi.

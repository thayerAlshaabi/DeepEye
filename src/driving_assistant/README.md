# DeepEye: DrivingAssistant

## Table of Contents
1. [Introduction](#introduction)
2. [Methods](#methods)


## Introduction
Our system consists of two major layers. In the first layer, a deep convolutional neural network mainly used for object detection and image classification. The network takes in a raw pixel image, then classify it into one of the labels defined in the dataset. The output of the CNN will then be fed into the second layer, which is going to perform online motion analysis and trajectory-based tracking to detect any situations that may pose a potential threat to the driving agent. Lastly, the result of our structure for motion (SfM) system will then be transformed to voice warning system that would notify the driver of any upcoming threats.

## Methods
Name | Description 
--- | ---
**user_interface** | This is a graphical user interface built using Python TkInter. For more information  please refer to [GUI](user_interface/README.md).
**set_prams** | set all parameters passed in from the user interface for the ObjectClassifier class.
**activate** | capture frames from the screen, and detect objects.

*Note:* press the (ESC) key to exit out of the loop.

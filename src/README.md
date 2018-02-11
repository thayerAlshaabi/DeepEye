# DeepEye: ObjectClassifier

## Table of Contents
1. [Introduction](#introduction)
2. [Methods](#methods)
3. [Customization](#customization)
4. [Licensing Information](#licensing-information)


## Introduction
This class is a deep convolutional neural network implemented using Tensorflow Object Detection API. The API has models trained on the [COCO dataset](http://cocodataset.org/#home) and [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). For more information about the API please refer to [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

We will be using this API for three purposes:
- Detecting relevant objects,
- Creating bounding boxes, and
- Classifying the objects (car, pedestrian, motorcycle, etc.)


## Methods
Name | Description 
--- | ---
**\_\_init\_\_** | The constructor doesn't necessarily require passing any parameters as they all have some satisfactory default values to start with. See [Customization](#customization) for detailed information. 
**download_model()** | Download pre-trained model from the tensorflow object detection API.
**load_model()** | Load pre-trained model into memory.
**setup()** | This method would download a trained model from the API if necessary files were not found. And, it loads the trained model into memory - preferably GPU memory using the methods stated above. Then, prep. the tensorflow computation graph and initiates a tensorflow session.
**scan_road()** | Capture frames and detecte objects.


## Customization
### Model Architecture
The API lists many different pre-trained models, training and hyperparameter tuning pipelines for (Faster R-CNN, SSD, and R-FCN) network meta-architectures coupled with various feature extractors like (Resnet-101, VGG-16, Inception v2-v3, and some others). By default, our implementation uses the (faster_rcnn_resnet101 | mAP=32) model. It's a fair compromise to achieve a relatively high mean average precision (mAP) without slowing down the model's performance too much. In other words, It provides the right speed/accuracy balance for our base model that would serve efficiently given our target platform and limited hardware.


**Note:** mAP is is the product of precision and recall on detecting bounding boxes. In short, the higher the mAP score, the more accurate the network is but that comes at the cost of processing speed.


To use a different classifier and/or different dataset you'll need to look at:

Parameter | Description 
--- | ---
**classifier_codename** | Codename of the pre-trained model used for object detection **([list of available models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))**.
**dataset_codename** | Codename of the dataset that the model was trained on (available datasets: **mscoco, kitti**).
**classifier_threshold** | The decision threshold : all detection scores below this given threshold will be discarded **(0.75 by default)**.

### Window Management
We use a cross-platform API for captureing screenshots called (Python MSS). For more information about MSS-API refer to [python-mss](http://python-mss.readthedocs.io/examples.html).

**Note:** By default our calss will capture the entire main monitor and pass it into the CNN.
Here's a list of additional parameters to look at:

Parameter | Description 
--- | ---
**monitor_id** | ID of the monitor to be captured.
**window_top_offset** | Top Offset in pixels **(0 by default)**.
**window_left_offset** | Left Offset in pixels **(0 by default)**.
**window_width** | The desired width of the captured window **(full width of the given monitor by default)**.
**window_height** | The desired height of the captured window **(full height of the given monitor by default)**.
**window_scale** | A scaling factor for the captured window **(1.0 by default)**.


## Licensing Information
The following code was adapted from the Tensorflow Object Detection API licensed under [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE)

Original SourceCode could be found @ [object_detection_tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

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
"""

# libraries and dependencies
# ---------------------------------------------------------------------------- #
from object_classifier.ObjectClassifier import *
from driving_assistant.warning_interface.WarningInterface import *
# ---------------------------------------------------------------------------- #


PEDESRIAN = 1

BIKES = [2, 3]

VEHICLES = [3, 6, 7, 8, 9]

TRAFFIC_LIGHT = 10

STOP_SIGN = 13

PARKING_METER = 14

ANIMALS = [17, 18, 19, 20, 21, 22, 23, 24, 25]


class DrivingAssistant:
    # Constructor
    def __init__(self):
        self.classifier = None
        #self.interface = Warning_Interface()
    
    def set_prams(self, classifier, dataset, threshold, visualization, lane_detection, monitor, top, left, width, height):

        if width !=0:
            self.classifier = ObjectClassifier(
                classifier_codename = classifier,
                dataset_codename = dataset,
                classifier_threshold = threshold,
                visualization = visualization,
                lane_detection = lane_detection,
                monitor_id = monitor,
                window_top_offset = top,
                window_left_offset = left,
                window_width = width,
                window_height = height,
            )
        else:
            self.classifier = ObjectClassifier(
                classifier_codename = classifier,
                dataset_codename = dataset,
                classifier_threshold = threshold,
                visualization = visualization,
                lane_detection = lane_detection,
                monitor_id = monitor,
                window_top_offset = top,
                window_left_offset = left,
            )

    def activate(self):
        #print("\n\n------------------------------------------\n\n")

        print("Activating DeepEye Advanced Co-pilot Mode")

        self.classifier.setup()
        #vp_start_warning_interface()
        
        while(True):
            self.classifier.scan_road()

            # get the detected objects by the neural network along with their confidence scores
            detected_objs = zip(self.classifier.detection_classes[0], self.classifier.detection_scores[0])
            
            # update warning interface as needed 
            for(obj_id, net_confidence) in detected_objs:
                
                if obj_id == PEDESRIAN and net_confidence >= self.classifier.classifier_threshold:
                    print("-- PEDESRIAN WARNING")
                
                elif obj_id == STOP_SIGN and net_confidence >= self.classifier.classifier_threshold:
                    print("-- STOP_SIGN WARNING")

                elif obj_id == TRAFFIC_LIGHT and net_confidence >= self.classifier.classifier_threshold:
                    print("-- TRAFFIC_LIGHT WARNING")

                elif obj_id in BIKES and net_confidence >= self.classifier.classifier_threshold:
                    print("-- BIKES WARNING")
                    #warningInterface.updateState()

                elif obj_id in ANIMALS and net_confidence >= self.classifier.classifier_threshold:
                    print("-- OBSTACLES WARNING")
                    #warningInterface.updateState()

            
            # Press ESC key to exit.
            if cv2.waitKey(25) & 0xFF == ord(chr(27)): # ESC=27 (ASCII)
                # Close all Python windows when everything's done
                cv2.destroyAllWindows()
                break
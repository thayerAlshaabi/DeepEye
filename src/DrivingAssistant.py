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
# ---------------------------------------------------------------------------- #

class DrivingAssistant:
    """

    """
    # Constructor
    def __init__(self):
        self.classifier = ObjectClassifier(
            classifier_codename = 'faster_rcnn_resnet101_coco_2017_11_08',
            dataset_codename = 'mscoco',
            classifier_threshold = .85,
            monitor_id = 1,
            window_width = 800,
            window_height = 640,
        )

    def activate(self):
        print("\n\n\n")
        print("  _____                  ______           ")
        print(" |  __ \                |  ____|          ")
        print(" | |  | | ___  ___ _ __ | |__  _   _  ___ ")
        print(" | |  | |/ _ \/ _ \ '_ \|  __|| | | |/ _ \\")
        print(" | |__| |  __/  __/ |_) | |___| |_| |  __/")
        print(" |_____/ \___|\___| .__/|______\__, |\___|")
        print("                  | |           __/ |     ")
        print("                  |_|          |___/      ")
        print("\n\n------------------------------------------\n\n")

        print("Activating DeepEye Advanced Co-pilot Mode")
        self.classifier.setup()

        while (True):
            self.classifier.scan_road()

            # Press ESC key to exit.
            if cv2.waitKey(25) & 0xFF == ord(chr(27)): # ESC=27 (ASCII)
                # Close all Python windows when everything's done
                cv2.destroyAllWindows()
                break

if __name__== "__main__":
    DeepEye = DrivingAssistant()
    DeepEye.activate()

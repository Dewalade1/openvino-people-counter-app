"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MODEL = "/home/workspace/model/frozen_inference_graph.xml"
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    print("[-] Connected to MQTT/Mosca server")
    return client

def draw_boxes(frame, result, args, width, height):
    """
    Draw bounding boxes onto the frame.
    """
    # Extract classes of images
    classes = cv2.resize(result[0].transpose((1,2,0)), (width, height), interpolation = cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    
    # Draw bounding boxes
    for box in result[0][0]:
        confidence = box[2]
        if confidence >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,225,255), 4)
            print("[-] Bounding box drawn to frame")
    return frame, unique_classes


def get_class_names(class_nums):
    class_names = []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names


def infer_on_video(args):
    network = Network()
    network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = network.get_input_shape()
    
    captured = cv2.VideoCapture(args.input)
    captured.open(args.input)
    
    width = int(captured.get(3))
    height = int(captured.get(4))
    
    output = cv2.VideoWriter("output_video.mp4", 0x00000021, 10, (width, height))
    
    while captured.isOpened():
        flag, frame = captured.read()
        if not flag:
            break
        key_pressed = cv2.waitkey(40)
        
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        network.async_inference(p_frame)
        
        if network.wait() == 0:
            result = network.extract_output()
            frame = draw_boxes(frame, result, args, width, height)
            output.write(frame)
        
        #Break if esc key is pressed
        if key_pressed == 27:
            break
            
    # Release the output writer, captured frames & destroy all openCV windows
    output.release()
    captured.release()
    cv2.destroyAllWindows()

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    captured = cv2.VideoCapture(args.input)
    captured.open(args.input)
    
    #Get shape actual of input
    width = int(captured.get(3))
    height = int(captured.get(4))

    ### TODO: Loop until stream is over ###
    while captured.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = captured.read()
        if not flag:
            break
        key_pressed = cv2.waitkey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.recize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.tramspose((2,0,1))
        p_frame = p_frame.reshape.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.async_inference(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            results = infer_network.extract_output()

            ### TODO: Extract any desired stats from the results ###
            output_frame, classes = draw_boxes(p_frame, results, args, width, height)
            class_names = get_class_names(classes)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            client.publish("person", json.dumps({"count" : count, "total" : total}))
            client.publish("person/duration", json.dumps({"duration" : duration}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(output_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        
        
        if key_pressed == 27:
            break
            
    captured.release()
    cv2.destroyAllWindows()
    
    #Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

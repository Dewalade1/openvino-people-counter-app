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

import numpy as np
from random import randint
from argparse import ArgumentParser
from inference import Network



# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


# Classification labels used by the model in the exact order
classes = ["Unknown", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant","street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]


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
    client.connect(MQTT_HOST,MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame, result, prob_threshold, width, height):
    """
    Draw bounding boxes to the frame
    
    Params
    frame: frame from camera/video
    result: list contains the result of inference
    
    return
    frame: frame with bounding box drawn on it
    """
    start_point = None
    end_point = None
    thickness = 5
    color = (255, 86, 0)
    
    for box in result[0][0]: 
        if box[2] > prob_threshold:
            start_point = (int(box[3] * width), int(box[4] * height))
            end_point = (int(box[5] * width), int(box[6] * height))
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            box_label = '{}: {:.2f}%'.format(classes[int(box[1])], box[2] * 100)
            frame = cv2.putText(frame, box_label , (int(box[3] * width)+ 5, int(box[4] * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 86, 0), 2)
            
    return frame

def person_count_per_frame(result, args):
    """
    Counts number of people in a frame
    
    params
    result: list contains the result of inference
    
    return
    count: count of number of people in a frame
    """
    count = 0
    
    for box in result[0][0]:
        confidence = box[2]
        prob_threshold = args.prob_threshold
        
        if confidence > prob_threshold:
            count += 1
    return count
    

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    params
    args: Command line arguments parsed by `build_argparser()`
    client: MQTT client
    
    :return: None
    """
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    network_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    single_image_mode = False
   
    image_extensions = ['.jpg','.bmp','.dpx', '.png', '.gif', '.webp', '.tiff', '.psd', '.raw', '.heif', '.indd']
    video_extensions = ['.mp4','.webm', '.mpg', '.mp2', '.mpeg', '.mpe', '.mpv', '.ogg', '.m4p', '.m4v', '.avi', '.wmv', '.mov', '.qt', '.flv', '.swf', '.avchd']
 
     # Check if the input is from webcam, an image, or a video
    if args.input == 'CAM':
        args.input = 0
        
    elif os.path.splitext(args.input)[1].lower() in video_extensions:
        output_file = 'output_video{}'.format(os.path.splitext(args.input)[1].lower())
        single_image_mode = False
        
    elif os.path.splitext(args.input)[1].lower() in image_extensions:
        output_file = "output_image{}".format(os.path.splitext(args.input)[1].lower())
        single_image_mode = True
        
    else:
        log.error(" [preprocessing block] File is not a video or an image in rasta format")
        log.error(" [preprocessing block] Please input an image in rasta format or video")
        exit(1)
    
    # Get and open video capture
    captured = cv2.VideoCapture(args.input)
    captured.open(args.input)
    
    #Get shape actual of input
    width = int(captured.get(3))
    height = int(captured.get(4))
    
    input_shape = network_input_shape['image_tensor']
    
    # Creates an output video if input file is a video
    if single_image_mode:
        output_video = None 
    else:
        path = 'outputs/videos/'
        output_video = cv2.VideoWriter(os.path.join(path , output_file), 0x00000021, 30, (width,height))

    report_count = 0
    count = 0
    prev_count = 0
    prev_duration = 0
    total_count = 0
    duration = 0
    
    ### TODO: Loop until stream is over ###
    while captured.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = captured.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor': p_frame, 'image_info': p_frame.shape[1:]}
        report_duration = None
        infer_start = time.time()
        infer_network.exec_net(request_id = 0, network_input = network_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            person_count = 0
            
            ### TODO: Get the results of the inference request ###
            infer_time_diff = time.time() - infer_start
            output = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            bounded_frame = draw_boxes(frame, output, prob_threshold, width, height)
            infer_time_text = "Inference time: {:.3f}ms".format(infer_time_diff * 1000)
            bounded_frame = cv2.putText(bounded_frame, infer_time_text, (15,15), cv2.FONT_HERSHEY_COMPLEX,0.45, (255, 86, 0), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            person_count = person_count_per_frame(output,args)
            
            if person_count != count:
                prev_count = count
                count = person_count
                
                if duration >= 3:
                    prev_duration = duration
                    duration = 0
                else:
                    duration += prev_duration
                    prev_duration = 0    #unknown, not needed
            else:
                duration += 1
                if duration >= 3:
                    report_count = count
                    if duration == 3 and count > prev_count:
                        total_count += count - prev_count
                    elif duration == 3 and count < prev_count:
                        report_duration = int((prev_duration/10.0) * 1000)
            
            
            client.publish("person", json.dumps({"count" : report_count, "total" : total_count}), qos = 0, retain = False)
            if report_duration is not None:
                client.publish("person/duration", json.dumps({"duration" : report_duration}), qos = 0, retain = False)

            ### TODO: Send the frame to the FFMPEG server ###
            bounded_frame = cv2.resize(bounded_frame, (width, height))
            sys.stdout.buffer.write(bounded_frame)
            sys.stdout.flush()

            ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                path = 'outputs/images/'
                cv2.imwrite(os.path.join(path , output_file ), bounded_frame)
            else:
                output_video.write(bounded_frame)
                
        
        # Break if esc key is pressed
        if key_pressed == 27:
            break
            
    # Release all captured frames and destroy any openCVwindows
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

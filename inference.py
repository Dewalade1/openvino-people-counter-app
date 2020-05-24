#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.network = None
        self.core = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        

    def load_model(self, model, device = "CPU", cpu_extension = None):
        '''
        Load the model given IR files. Defaults device = CPU.
        Asynchronous requests made within.
        '''
        ### TODO: Load the model ###        
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the core
        self.core = IECore()
        
        # Read the IR as an IENetwork
        self.network = IENetwork(model = model_xml, weights = model_bin)
        
        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(self.network, device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        ### TODO: Add any necessary extensions ###
        # Add cpu extension if any unsupported layers are found
        if (len(unsupported_layers) != 0) and (cpu_extension and "CPU" in device):
            self.core.add_extension(cpu_extension, device)
            
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network = self.core.load_network(self.network, device)
        
        # Get input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.exec_network

    def get_input_shape(self):
        '''
        This gets the input shape of the network
        '''
        ### TODO: Return the shape of the input layer ###
        input_shapes = {}
        for network_input in self.network.inputs:
            input_shapes[network_input] = (self.network.inputs[network_input].shape)
        return input_shapes

    def exec_net(self, network_input, request_id):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.exec_network.start_async(request_id = 0, inputs = network_input)
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.infer_request_handle.wait()
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
    
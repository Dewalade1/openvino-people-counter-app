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

# MODEL = "/home/workspace/model/frozen_inference_graph.xml"
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


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

        self.core = IECore()
        self.network = IENetwork(model = model_xml, weights = model_bin)

        ### TODO: Check for supported layers ###
        supported_layers = core.query_network(self.network, device)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("[Inference Engine] Unsupported layers found: \n {}".format(unsupported_layers))
            print("[Inference Engine] Check to see if unsupported extensions are available to IECore")
            print("[Inference Engine] If not available, add as custom layers")
            exit(1)

        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.core.addextension(cpu_extension, device)

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###

        self_exec_network = self.core.load_network(self.network, device)
        print("[Inference Engine] IR successfully loaded into Inference Engine")

        # Get input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        '''
        This gets the input shape of the network
        '''
        ### TODO: Return the shape of the input layer ###
        input_shape = self.network.inputs[self.input_blob].shape
        print("[Inference Engine] Input shape found!")
        return input_shape

    def exec_net(self, frame):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id = 0, inputs={self.input_blob: image})

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)

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
        

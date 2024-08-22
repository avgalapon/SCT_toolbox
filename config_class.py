#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   config_class.py
@Time    :   2024/03/19 10:36:25
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   Config class for CAPTAIN sCT Module
             Read the parameter json file and store the parameters
'''

import os
import json

class Config:
    def __init__(self, json_parameters_path):
        self.params = self.read_json_file(json_parameters_path)
        
        self.type = self.params['type']
        
        # Directory Paths
        self.data_path = self.params['data_path']
        self.input_path = self.params['input_path']
        self.groundtruth_path = self.params['groundtruth_path'] # from CAPTAIN
        self.output_path = self.params['output_path'] # from CAPTAIN - same as datapath
        self.temp_preprocessed_path = self.params['temp_preprocessed_path'] # fixed if possible - will be deleted
        self.weights_path = self.params['weights_path'] # variable - depends on the model
        
        # Filenames
        self.fname = self.params['fname']
        
        # Data parameters
        self.img_type = self.params['img_type']
        self.X = self.params['X']
        self.Y = self.params['Y']
        self.VOI = self.params['VOI']
        self.orientation = self.params['orientation']
        
        # Model parameters
        self.model_type = self.params['model_type']
        
        # # Synthesis parameters
        self.device = self.params['device']       
        self.dropout_enable = self.params['dropout_enable']     
        self.dropout_rate = self.params['dropout_rate'] 
        
        # Evaluation parameters
        self.DIR = self.params['DIR']
        self.EVAL = self.params['EVAL']
        
        # Training parameters
        self.epoch = self.params['epoch']
        self.learning_rate = self.params['learning_rate']
        self.decay_epoch = self.params['decay_epoch']
        self.lambda_decay = self.params['lambda_decay']
        
        assert self.type in ['train', 'synthesis']
        assert self.img_type in ['CBCT', 'MRI'], "Image type not recognized"
        assert self.model_type in ['DCNN', 'cycleGAN', 'cGAN'], "Model type not recognized"
        assert os.path.isdir(self.output_path), f"Output path {self.output_path} does not exist"
        # assert os.path.isdir(self.weights_path), f"Weights path {self.weights_path} does not exist"
        assert isinstance(self.DIR, bool), "DIR must be a boolean"
        assert isinstance(self.EVAL, bool), "EVAL must be a boolean"
        assert isinstance(self.dropout_enable, bool), "dropout_enable must be a boolean"
        assert isinstance(self.epoch, int), "epoch must be an integer"
        assert isinstance(self.decay_epoch, int), "decay_epoch must be an integer"
        assert isinstance(self.lambda_decay, float), "lambda_decay must be a float"
        assert isinstance(self.learning_rate, float), "learning_rate must be a float"
        assert isinstance(self.dropout_rate, float), "dropout_rate must be a float"    

    @staticmethod
    def read_json_file(path):
        with open(path, 'r') as f:
            return json.load(f)
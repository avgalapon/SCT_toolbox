#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   model_class.py
@Time    :   2024/05/24 16:49:27
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''
import torch
import model.DCNN_networks as DCNN_networks
import model.cycleGAN_networks as cycleGAN_networks
import model.cGAN_networks as cGAN_networks
from utils import enable_dropout

class Model:
    def __init__(self, config):
        self.config = config
    
    def initiate_device(self):
        return torch.device(f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu")
    
    def define_models(self):
        models_dict = {'DCNN': DCNN_networks.Generator(1,64,0.1),
                       'cycleGAN': cycleGAN_networks.define_G(1,1,64,"resnet_9blocks",'instance',True),
                       'cGAN': cGAN_networks.define_G(1, 1, 64, "resnet_9blocks", 'instance', True)}
        return models_dict
    
    def define_training_models(self):
        models_dict = {'DCNN': DCNN_networks.Generator(1,64,0.1),
                       'cycleGAN_A2B': cycleGAN_networks.define_G(1,1,64,"resnet_9blocks",'instance',use_dropout=True),
                       'cycleGAN_B2A': cycleGAN_networks.define_G(1,1,64,"resnet_9blocks",'instance',use_dropout=True),
                       'cycleGAN_DA': cycleGAN_networks.define_D(1,64,"basic",norm='instance'),
                       'cycleGAN_DB': cycleGAN_networks.define_D(1,64,"basic", norm='instance'),
                       'cGAN_G': cGAN_networks.define_G(1, 1, 64, "resnet_9blocks", 'instance', use_dropout=True),
                       'cGAN_D': cGAN_networks.define_D(1, 64, "basic", norm='instance')}
        return models_dict
    
    def load_weights(self, model, weights_path):
        print(f'loading weights from {weights_path}...' )
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        return model
    
    def initialize_models(self):
        device = self.initiate_device()
        models_dict = self.define_models()
        model = models_dict[self.config.model_type].to(device)
        model = self.load_weights(model, self.config.weights_path)
        return model, device
    
    def intialize_training_models(self):
        device = self.initiate_device()
        models_dict = self.define_training_models()
        initialized_model = {}
        if self.config.model_type == 'DCNN':
            initialized_model['DCNN'] = models_dict['DCNN'].to(device)
        elif self.config.model_type == 'cycleGAN':
            initialized_model['cycleGAN_A2B'] = models_dict['cycleGAN_A2B'].to(device)
            initialized_model['cycleGAN_B2A'] = models_dict['cycleGAN_B2A'].to(device)
            initialized_model['cycleGAN_DA'] = models_dict['cycleGAN_DA'].to(device)
            initialized_model['cycleGAN_DB'] = models_dict['cycleGAN_DB'].to(device)
        elif self.config.model_type == 'cGAN':
            initialized_model['cGAN_G'] = models_dict['cGAN_G'].to(device)
            initialized_model['cGAN_D'] = models_dict['cGAN_D'].to(device)
        else:
            raise ValueError('Model type not recognized')
        return initialized_model, device
    
    # 
    
    def enable_dropout(self, model):
        # Enable dropout
        model.eval()
        if self.config.dropout_enable:
            enable_dropout(model)
            print('Dropout Enabled')
            num_inference = 10
        else: 
            num_inference = 1
        return num_inference
    
    
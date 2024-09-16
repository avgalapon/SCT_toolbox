#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/07/18 10:38:40
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''

import os
from config_class import Config
import preprocessor as Preprocess
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import prepare_data_nrrd_class
import model_class
import losses
import train_class

current_dir = os.path.dirname(os.path.abspath(__file__))

json_path = fr"{current_dir}/config_files/train_cGAN_MR.json"
config = Config(json_path)

patient_list = sorted(os.listdir(config.data_path))
print(patient_list)

train_data, val_data = train_test_split(patient_list, test_size=0.3, random_state=42)
print("Total number of patients: ", len(patient_list))
print("Number of training patients: ", len(train_data))
print("Number of validation patients: ", len(val_data))

prep = Preprocess.Preprocessor_train(config)
prep.preprocess_train(train_data, train_data=True)
prep.preprocess_train(val_data, train_data=False)

print("Loading dataset to dataloader")
prepare_data = prepare_data_nrrd_class.PrepareDataset(config)
train_dataloader, valid_dataloader = prepare_data.create_train_dataset()

model, device = model_class.Model(config).intialize_training_models()
loss = losses.define_loss(config)

train_class.Train(config, model, device, loss, train_dataloader, valid_dataloader, plot=True).train_DCNN()
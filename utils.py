#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/07/15 14:11:21
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''

import json
import os
import SimpleITK as sitk

def read_json(file_path):
    """
    Read a JSON file and return the data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    """
    Write data to a JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def modify_json(file_path, key, new_value):
    """
    Modify the value of a specific key in a JSON file.
    """
    data = read_json(file_path)
    if key in data:
        print(f"Old value of '{key}': {data[key]}")
        data[key] = new_value
        print(f"New value of '{key}': {data[key]}")
        write_json(file_path, data)
    else:
        print(f"Key '{key}' not found in the JSON file.")
        
def enable_dropout(model):
    """Enables dropout during inference"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
        
def return_to_HU_cycleGAN(normX, max_X=3000.0, min_X=-1024.0):
    minmax = max_X - min_X
    X = (normX*minmax) + min_X
    return X

def return_to_HU_cGAN(normX, CT_max=3000.0, CT_min=-1024.0):
    return ((normX + 1) / 2) * (CT_max - CT_min) + CT_min
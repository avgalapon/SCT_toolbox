#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   generate_sct_bash.py
@Time    :   2024/06/25 10:11:16
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''

import json
import os
import fnmatch
import SimpleITK as sitk
from utils import modify_json
from evaluation_class import Evaluate_sCT

current_dir = os.path.dirname(os.path.abspath(__file__))

json_path = fr"{current_dir}/config_files/tp_testMR.json"
datapath = fr'/data/galaponav/dataset/newHN_MR_test'

model_list = ['cGAN'] # run for DCNN after for FX and FT
patient_list = sorted(os.listdir(datapath))[:1]
print(patient_list)

for patient in patient_list:
    exception_list = []
    for model in model_list:
        input_path = fr"/data/galaponav/dataset/newHN_MR_test/{patient}/MRI"
        groundtruth_path = fr"/data/galaponav/dataset/newHN_MR_test/{patient}/CT"
        
        input_list = os.listdir(input_path)
        groundtruth_list = fnmatch.filter(os.listdir(groundtruth_path),'*pCT*')
        
        new_inputpath = os.path.join(input_path, input_list[0])
        new_groundtruthpath = os.path.join(groundtruth_path, groundtruth_list[0])
        
        new_datapath = fr"/data/galaponav/dataset/newHN_MR_test/{patient}"
        new_outputpath = fr"/data/galaponav/dataset/newHN_MR_test/{patient}"
        fname = f"{patient}_MR"            

        modify_json(json_path, 'output_path', new_outputpath)
        modify_json(json_path, 'input_path', new_inputpath)
        modify_json(json_path, 'groundtruth_path', new_groundtruthpath)
        modify_json(json_path, 'fname', fname)
        modify_json(json_path, 'model_type', model)
        
        cmd = f"python /home/galaponav/art/scripts/PhD/SCT_toolbox/CTsynthesizer.py {json_path}"
        os.system(cmd)
        print(f"{patient} sCT generated")
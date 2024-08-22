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
import shutil
import os
import fnmatch
import SimpleITK as sitk
from utils import modify_json
from evaluation_class import Evaluate_sCT

current_dir = os.path.dirname(os.path.abspath(__file__))

json_path = fr"{current_dir}/config_files/tp_testMR.json"
datapath = fr'/data/galaponav/dataset/newHN_MR_test'

model_list = ['DCNN'] # run for DCNN after for FX and FT
patient_list = sorted(os.listdir(datapath))[:1]
print(patient_list)
list_suffix = ['_single_noEval']


for patient in patient_list:
    exception_list = []
    for model in model_list:
        for sufff in list_suffix:
            
            input_path = fr"/data/galaponav/dataset/newHN_MR_test/{patient}/MRI"
            groundtruth_path = fr"/data/galaponav/dataset/newHN_MR_test/{patient}/CT"
            
            input_list = os.listdir(input_path)
            groundtruth_list = fnmatch.filter(os.listdir(groundtruth_path),'*pCT*')
            
            new_inputpath = os.path.join(input_path, input_list[0])
            new_groundtruthpath = os.path.join(groundtruth_path, groundtruth_list[0])
            
            new_datapath = fr"/data/galaponav/dataset/newHN_MR_test/{patient}"
            new_outputpath = fr"/data/galaponav/dataset/newHN_MR_test/{patient}"
            fname = f"{patient}_MR{sufff}"            
            
            if sufff in ['_dropout_noEval', '_dropout_Eval', '_dropout_DIR', '_dropout_DIR_Eval']:
                modify_json(json_path, 'dropout_enable', True)
            else:
                modify_json(json_path, 'dropout_enable', False)
                
            if sufff in ['_single_noEval', '_dropout_noEval']:
                modify_json(json_path, 'EVAL', False)
            else:
                modify_json(json_path, 'EVAL', True)
                
            if sufff in ['_single_DIR', '_dropout_DIR']:
                modify_json(json_path, 'DIR', True)
            else:
                modify_json(json_path, 'DIR', False)

            modify_json(json_path, 'output_path', new_outputpath)
            modify_json(json_path, 'input_path', new_inputpath)
            modify_json(json_path, 'groundtruth_path', new_groundtruthpath)
            modify_json(json_path, 'fname', fname)
            modify_json(json_path, 'model_type', model)
            
            cmd = f"python /home/galaponav/art/scripts/PhD/SCT_toolbox/CTsynthesizer.py {json_path}"
            os.system(cmd)
            print(f"{patient} sCT generated")
        
            # delete all files in delete_list except exception_list
            exception_list = [f"sCT_{model}_{patient}_MR_single_noEval.nrrd", f"sCT_{model}_{patient}_MR_dropout_noEval.nrrd", f"sCT_{model}_{patient}_MR_single_Eval.nrrd", f"sCT_{model}_{patient}_MR_dropout_Eval.nrrd", f"sCT_{model}_{patient}_MR_single_DIR.nrrd", f"sCT_{model}_{patient}_MR_dropout_DIR.nrrd", f"sCT_{model}_{patient}_MR_single_DIR_Eval.nrrd", f"sCT_{model}_{patient}_MR_dropout_DIR_Eval.nrrd"]

            # Only delete .nrrd files except those in exception_list
            delete_list = [f for f in os.listdir(new_outputpath) if f.endswith('.nrrd')]
            exception_set = set(exception_list)  # Use a set for faster lookup
            print(f"Exception set: {exception_set}")

            # Iterate through the delete list and remove files not in thse exception list
            for item in delete_list:
                if item not in exception_set:
                    file_path = os.path.join(new_outputpath, item)
                    print(f"Deleting: {file_path}")
                    try:
                        # Using shutil to delete the file
                        shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")
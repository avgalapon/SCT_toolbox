#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   CTevaluator.py
@Time    :   2024/07/15 14:05:22
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''

import os
import time
import argparse
import shutil
import model_class
import prepare_data_class
import prepare_data_nrrd_class
from config_class import Config
import preprocessor as Preprocess
from synthesis_class import Generate_sCT
from evaluation_class import Evaluate_sCT

# json_path = r"/home/galaponav/art/scripts/PhD/generate_sct_paper3/tp_test.json"
#    "weights_path": "/data/galaponav/output/newHN_CBCT/DCNN/weights/epoch_18_UNET_forCap_best.pth",

def main():
    parser = argparse.ArgumentParser(description='Generate sCT image')
    parser.add_argument("json_parameters_path", help="Path to the test parameters file")
    args = parser.parse_args()
    cfg = Config(args.json_parameters_path)
            
    # Evaluate sCT
    EvaluateIQ = Evaluate_sCT(cfg)
    EvaluateIQ.run_IQeval(250)

if __name__ == "__main__":
    main()


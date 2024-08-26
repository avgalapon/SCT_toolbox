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
import SimpleITK as sitk
from utils import modify_json, return_to_HU_cGAN, return_to_HU_cycleGAN


json_path = fr"/home/galaponav/art/scripts/PhD/SCT_toolbox/config_files/tp_test.json"
datapath = fr'/data/galaponav/dataset/newHN_CBCT_test'


# TLtype = 'FX'
model_list = ['cycleGAN'] # run for DCNN after for FX and FT
# weights_list = [f'netGA2B_TL05_FX.pth',f'netGA2B_TL10_FX.pth',f'netGA2B_TL15_FX.pth',f'netGA2B_TL20_FX.pth',f'netGA2B_TL25_FX.pth',f'netGA2B_TL30_FX.pth',f'netGA2B_TL35_FX.pth',f'netGA2B_TL40_FX.pth',f'netG_A2B_new_70.pth', f'netG_A2B_old_66.pth']
patient_list = sorted(os.listdir(datapath))

# TLtype_list = ['FX','FT']
TLtype_list = ['FX']
for TLtype in TLtype_list:
    if TLtype == 'FX':
        weightspath = fr'/data/galaponav/output/paper2/weights/cycleGAN_featureX'
        # weights_list = [f'netGA2B_TL05_FX.pth',f'netGA2B_TL10_FX.pth',f'netGA2B_TL15_FX.pth',f'netGA2B_TL20_FX.pth',f'netGA2B_TL25_FX.pth',f'netGA2B_TL30_FX.pth',f'netGA2B_TL35_FX.pth',f'netGA2B_TL40_FX.pth',f'netG_A2B_new_70.pth', f'netG_A2B_old_66.pth']
        # fname_list = [f'_TL05_{TLtype}', f'_TL10_{TLtype}', f'_TL15_{TLtype}', f'_TL20_{TLtype}', f'_TL25_{TLtype}', f'_TL30_{TLtype}', f'_TL35_{TLtype}', f'_TL40_{TLtype}', f'_new', f'_old']
        weights_list = [f'netGA2B_TL40_FX2.pth']
        fname_list = [f'_TL40_{TLtype}']
    elif TLtype == 'FT':
        weightspath = fr'/data/galaponav/output/paper2/weights/cycleGAN_finetune'
        weights_list = [f'netGA2B_TL05_FT.pth',f'netGA2B_TL10_FT.pth',f'netGA2B_TL15_FT.pth',f'netGA2B_TL20_FT.pth',f'netGA2B_TL25_FT.pth',f'netGA2B_TL30_FT.pth',f'netGA2B_TL35_FT.pth', f'netG_A2B_new_70.pth', f'netG_A2B_old_66.pth']
        fname_list = [f'_TL05_{TLtype}', f'_TL10_{TLtype}', f'_TL15_{TLtype}', f'_TL20_{TLtype}', f'_TL25_{TLtype}', f'_TL30_{TLtype}', f'_TL35_{TLtype}']  

    for i,model in enumerate(model_list):
        for j,weights in enumerate(weights_list):
            weights_path = fr"{weightspath}/{weights_list[j]}"
            modify_json(json_path, 'weights_path', weights_path)

            for patient in patient_list:
                new_datapath = fr"/data/galaponav/dataset/newHN_CBCT_test/{patient}"
                new_outputpath = fr"/data/galaponav/dataset/newHN_CBCT_test/{patient}"
                fname = f"{patient}{fname_list[j]}"

                modify_json(json_path, 'data_path', new_datapath)
                modify_json(json_path, 'output_path', new_outputpath)
                modify_json(json_path, 'fname', fname)
                modify_json(json_path, 'model_type', model)
                
                cmd = f"python /home/galaponav/art/scripts/PhD/SCT_toolbox/CTsynthesizer.py {json_path}"
                os.system(cmd)
                print(f"{patient} sCT generated")
                
                # Run when using cycleGAN/cGAN because sCT output is normalized
                sct_path = os.path.join(datapath, patient, f'sCT_{model}_{fname}.nrrd')
                sct = sitk.ReadImage(sct_path)
                sct_arr = sitk.GetArrayFromImage(sct)
                
                if model == 'cycleGAN':
                    sct_arr = return_to_HU_cycleGAN(sct_arr)
                elif model == 'cGAN':
                    sct_arr = return_to_HU_cGAN(sct_arr)
                else:
                    print("No de-normalization needed")
                
                sct_img = sitk.GetImageFromArray(sct_arr)
                sct_img.CopyInformation(sct)
                sitk.WriteImage(sct_img, sct_path)
                print(f"{patient} sCT unnormalized")
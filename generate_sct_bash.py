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


json_path = fr"/home/galaponav/art/scripts/PhD/SCT_toolbox/tp_test2.json"
datapath = fr'/data/galaponav/dataset/newHN_CBCT_test'


# TLtype = 'FX'
model_list = ['DCNN'] # run for DCNN after for FX and FT
patient_list = sorted(os.listdir(datapath))
print(patient_list)

TLtype_list = ['FX','FT']
for TLtype in TLtype_list:
    if TLtype == 'FX':
        weightspath = fr'/data/galaponav/output/paper2/weights/DCNN_featureX'
        weights_list = [f'UNET_TL05_e147.pth',f'UNET_TL10_e34.pth',f'UNET_TL15_e81.pth',f'UNET_TL20_e42.pth',f'UNET_TL25_e113.pth',f'UNET_TL30_e75.pth',f'UNET_TL35_e22.pth',f'UNET_TL40_e63.pth', f'UNET_new_102.pth', f'DCNN_old187.pth'] # FX
        fname_list = [f'_TL05_{TLtype}', f'_TL10_{TLtype}', f'_TL15_{TLtype}', f'_TL20_{TLtype}', f'_TL25_{TLtype}', f'_TL30_{TLtype}', f'_TL35_{TLtype}', f'_TL40_{TLtype}', f'_new', f'_old']
    elif TLtype == 'FT':
        weightspath = fr'/data/galaponav/output/paper2/weights/DCNN_finetune'
        weights_list = [f'UNET_TL05_e80.pth',f'UNET_TL10_e149.pth',f'UNET_TL15_e55.pth',f'UNET_TL20_e63.pth',f'UNET_TL25_e29.pth',f'UNET_TL30_e53.pth',f'UNET_TL35_e26.pth',f'UNET_TL40_e102.pth', f'UNET_new_102.pth', f'DCNN_old187.pth'] # FT
        fname_list = [f'_TL05_{TLtype}', f'_TL10_{TLtype}', f'_TL15_{TLtype}', f'_TL20_{TLtype}', f'_TL25_{TLtype}', f'_TL30_{TLtype}', f'_TL35_{TLtype}', f'_TL40_{TLtype}', f'_new', f'_old']

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
                
                cmd = f"python /home/galaponav/art/scripts/PhD/SCT_toolbox/CTsynthesizer_nrrd.py {json_path}"
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


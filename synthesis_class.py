#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   synthesis_class.py
@Time    :   2024/05/24 17:05:44
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''
import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import model_class
from torch.cuda.amp import autocast, GradScaler  # Import AMP modules
from utils import return_to_HU_cycleGAN, return_to_HU_cGAN
import prepare_data_nrrd_class as prepare_data_class

class Generate_sCT:
    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.dropout_enable = config.dropout_enable
        
    def inference_loop(self):            
        progressbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        generated_B = []
        generated_b = []

        # Use AMP for faster processing if available
        scaler = GradScaler()
        
        if self.config.model_type != 'cGAN':
            for i, data in progressbar:
                real_X = data['A'].to(self.device)
                
                with autocast():  # Use mixed precision
                    # Generate output
                    fake_B = self.model(real_X)
                
                generated_B.append(fake_B.cpu().detach().numpy())
                progressbar.set_description(f"Processing slice {i+1} of {len(self.dataloader)}")

            # Get numpy array from tensor for real CT (B) (or MRI (A)) and generated CT
            ct_gen = np.squeeze(generated_B)
            unc_gen = np.zeros_like(ct_gen)
            
        else:        
            for i, data in progressbar:
                real_X = data['A'].to(self.device)
                mask = data['mask'].to(self.device)
                
                with autocast():  # Use mixed precision
                    # Generate output
                    if mask.max() > 0:
                        fake_B, unc_b = self.model(real_X)
                    else:
                        fake_B = torch.full_like(real_X, -1)
                        unc_b = torch.full_like(real_X, -1)
                    
                generated_B.append(fake_B.cpu().detach().numpy())
                generated_b.append(unc_b.cpu().detach().numpy())
                progressbar.set_description(f"Processing slice {i+1} of {len(self.dataloader)}")
            
            ct_gen = np.squeeze(generated_B)
            unc_gen = np.squeeze(generated_b)
                
        return ct_gen, unc_gen
    
    def save_generated_sct(self, predictions):
        print("Stacking slices...")
        sct_stack = np.stack(predictions)
        sct_mean = np.mean(sct_stack, axis=0)
        sct_variance = np.var(sct_stack, axis=0)
        reference_input = sitk.ReadImage(os.path.join(self.config.output_path, self.config.X))
        
        sct_img = np.squeeze(sct_mean)
        unc_img = np.squeeze(sct_variance)
        
        if self.config.img_type == 'MRI':
            sct_img = self.revert_image(sct_img, reference_input)
            unc_img = self.revert_image(unc_img, reference_input)
        
        print('Updating image properties...')

        sct_img.CopyInformation(reference_input)
        unc_img.CopyInformation(reference_input)
        
        # Denormalizing sCT
        sct_img = self.unnorm_sct(sct_img)
        unc_img = self.unnorm_sct(unc_img)
        
        print('Saving sCT-nrrd...')
        sct_fname = os.path.join(self.config.output_path, f'sCT_{self.config.model_type}_{self.config.fname}.nrrd')
        sitk.WriteImage(sct_img,sct_fname, True)
        
        if self.dropout_enable:
            unc_fname = os.path.join(self.config.output_path, f'epistemic_{self.config.model_type}_{self.config.fname}.nrrd')
            sitk.WriteImage(unc_img,unc_fname, True)         
            print("File saved in: %s and %s" % (sct_fname, unc_fname))
        else:
            print("File saved in: %s" % (sct_fname))
        
    def revert_image(self, img, img_ref):
        img_ref = sitk.GetArrayFromImage(img_ref)
        original_shape = img_ref.shape
        
        max_image_shape = 512
        offsets = (int(np.floor(max_image_shape-original_shape[0])/2.0), int(np.floor(max_image_shape-original_shape[1])/2.0), int(np.floor(max_image_shape-original_shape[2])/2.0))
        
        original_img = img[
            offsets[0]:offsets[0] + original_shape[0],
            offsets[1]:offsets[1] + original_shape[1],
            offsets[2]:offsets[2] + original_shape[2]]
        
        return sitk.GetImageFromArray(original_img.astype(np.float32))
    
    def unnorm_sct(self, sct_arr):
        if self.config.model_type == 'cycleGAN':
            sct_arr = return_to_HU_cycleGAN(sct_arr)
        elif self.config.model_type == 'cGAN':
            sct_arr = return_to_HU_cGAN(sct_arr)
        else:
            pass
        return sct_arr
    
    def save_data_uncertainty(self, uncertainty):
        print("Stacking slices...")
        variance = [u**2 for u in uncertainty]
        combined_var = np.mean(variance, axis=0)
        combined_std = np.sqrt(combined_var)
        
        unc_img = sitk.GetImageFromArray(np.squeeze(combined_std))
        reference_cbct = sitk.ReadImage(os.path.join(self.config.output_path, self.config.X))
        unc_img.CopyInformation(reference_cbct)
        unc_img = self.unnorm_sct(unc_img)
        unc_fname = os.path.join(self.config.output_path, f'aleatoric_{self.config.model_type}_{self.config.fname}.nrrd')
        sitk.WriteImage(unc_img,unc_fname, True)         
        print("File saved in: %s" % (unc_fname))
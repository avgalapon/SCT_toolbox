#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   evaluation_class.py
@Time    :   2024/07/15 13:32:34
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''


import os
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import copy

class Evaluate_sCT:
    def __init__(self, config):
        self.config = config
        self.IQdict =  {}
        
    def load_images(self):
        print(f"Loading images for patient: {self.config.fname}")
        path_Ground_CT = os.path.join(self.config.output_path, self.config.Y)
        path_sCT = os.path.join(self.config.output_path, f'sCT_{self.config.model_type}_{self.config.fname}.nrrd')
        path_mask = os.path.join(self.config.output_path, self.config.VOI)
        
        if not os.path.isfile(path_Ground_CT):
            raise FileNotFoundError(f"Ground truth CT is required for evaluation. File not found: {path_Ground_CT}" )
        
        Ground_CT = sitk.GetArrayFromImage(sitk.ReadImage(path_Ground_CT))
        sCT = sitk.GetArrayFromImage(sitk.ReadImage(path_sCT))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(path_mask))
        
        return Ground_CT, sCT, mask
    
    def mean_absolute_error(self, Ground_CT, sCT, mask):        
        # Calculate MAE
        diff = np.abs(Ground_CT - sCT)
        mae = np.mean(diff[mask>0])
        return mae
    
    def PSNR(self, Ground_CT, sCT, mask):       
        # Calculate PSNR
        mse = np.mean((Ground_CT[mask>0] - sCT[mask>0])**2)
        psnr = 20*np.log10(1024/np.sqrt(mse))
        return psnr
    
    def DICE(self, Ground_CT_bones, sCT_bones):
        intersection = np.sum(Ground_CT_bones * sCT_bones)
        dice = 2*intersection / (np.sum(Ground_CT_bones) + np.sum(sCT_bones))
        
        return dice
    
    def segment_bones(self, GCT_img, sCT_img,  bone_threshold=250):
        CT_bones = copy.copy(GCT_img)
        CT_bones[CT_bones < bone_threshold] = 0
        CT_bones[CT_bones >= bone_threshold] = 1
        
        sCT_bones = copy.copy(sCT_img)
        sCT_bones[sCT_bones < bone_threshold] = 0
        sCT_bones[sCT_bones >= bone_threshold] = 1
        
        return CT_bones, sCT_bones
        
    def SSIM(self, Ground_CT, sCT):
        # Convert to numpy array
        data_range = Ground_CT.max() - Ground_CT.min()
        
        ssim_score = ssim(sCT, Ground_CT, data_range=data_range)
        return ssim_score
    
    def image_quality_metrics(self, Ground_CT, sCT, mask, dict_path):
        print(f"Calculating Image Quality Metrics for patient: {self.config.fname}")
        # Create Dictionary for Image Quality Metrics
        IQ_dict = {}
                
        # Calculate MAE
        mae = self.mean_absolute_error(Ground_CT, sCT, mask)
        
        # Calculate PSNR
        psnr = self.PSNR(Ground_CT, sCT, mask)
        
        # Calculate SSIM
        ssim = self.SSIM(Ground_CT, sCT)
        
        # Calculate DICE
        GCT_bones, sCT_bones = self.segment_bones(Ground_CT, sCT)
        dice = self.DICE(GCT_bones, sCT_bones)
        
        # Add to dictionary
        IQ_dict['Patient'] = fr"{self.config.model_type}_{self.config.fname}"
        IQ_dict['MAE'] = mae
        IQ_dict['PSNR'] = psnr
        IQ_dict['SSIM'] = ssim
        IQ_dict['DICE'] = dice
        
        # Save dictionary to dict_path
        print(f"Saving Image Quality Metrics to: {dict_path}")
        np.save(dict_path, IQ_dict)
        
        # Save as txt
        with open(dict_path.replace('.npy', '.txt'), 'w') as f:
            for key in IQ_dict.keys():
                f.write(f"{key}: {IQ_dict[key]}\n")
        
        return mae, psnr, ssim, dice
        
    def save_plots(self, Ground_CT, sCT, idx):
        print('Generating MAE plot...')
        # Image Difference
        diff_image = np.abs(Ground_CT - sCT)
        
        # Segmented bones
        GCT_bones, sCT_bones = self.segment_bones(Ground_CT, sCT)
        
        plt.figure(figsize=(15,15))
        plt.subplot(131)
        plt.imshow(diff_image[:,:,idx], cmap='RdBu_r', origin='lower')
        plt.title('Absolute Difference')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(132)
        plt.imshow(GCT_bones[:,:,idx], cmap='gray', origin='lower')
        plt.title('Ground Truth Bones')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(sCT_bones[:,:,idx], cmap='gray', origin='lower')
        plt.title('sCT Bones')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_path, f'IQ_{self.config.model_type}_{self.config.fname}.png'))
        plt.close()
        
    def run_IQeval(self, idx):
        # Load Images
        Ground_CT, sCT, mask = self.load_images()
        
        # Save Image Quality Metrics
        dict_path = os.path.join(self.config.output_path, f'IQ_{self.config.model_type}_{self.config.fname}.npy')
        mae, psnr, ssim, dice, = self.image_quality_metrics(Ground_CT, sCT, mask, dict_path)
        
        # Save Plots
        self.save_plots(Ground_CT, sCT, idx)
        
        # Print Image Quality Metrics
        print(f"Patient: {self.config.fname}")
        print(f"MAE: {mae}")
        print(f"PSNR: {psnr}")
        print(f"SSIM: {ssim}")
        print(f"DICE: {dice}")
        
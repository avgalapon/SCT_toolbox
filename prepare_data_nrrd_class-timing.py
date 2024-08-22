#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_data_class.py
@Time    :   2024/05/24 16:41:32
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''
import os
import fnmatch
import pre_process_tools
import torch.utils.data
from ImageDataset import ImageDataset, ImageDataset_train
import itk
import SimpleITK as sitk
import time

class prepare_dataset:
    def __init__(self,config, reference_MR=None):
        print("Preparing images...")
        self.config = config
        self.img_type = config.img_type
        self.reference_MR = reference_MR
        self.type = config.type
        self.current_dir = os.path.dirname(__file__)
                      
        if self.img_type == 'CBCT':
            param_path = os.path.join(self.current_dir, 'registration_parameters/parameters_CBCT_brain.txt')    
        else:
            param_path = os.path.join(self.current_dir, 'registration_parameters/parameters_MR.txt')
            
        self.registration_parameters = pre_process_tools.read_parameter_map(param_path)
    
    def run_sitk(self, topslice, bottomslice, DIR=False, eval=True):
        time_elapsed_conversion = 0
        time_elapsed_rigid = 0
        time_elapsed_bcorr = 0
        time_elapsed_histmatch = 0
        time_elapsed_DIR = 0        
        
        time_start_conversion = time.time()
        
        input_dir = self.config.input_path        
        input_nrrd = os.path.join(self.config.output_path,f"{self.img_type}.nrrd")
        
        if not os.path.isfile(input_nrrd):
            print("Converting CBCT dicom to NRRD...")
            pre_process_tools.convert_dicom_to_nifti(input_dir, input_nrrd)
            self.remove_tb_slices(topslice, bottomslice, img_type=self.img_type)
            
        mask = os.path.join(self.config.output_path,'mask.nrrd')
        print("Segmenting Mask...")
        pre_process_tools.segment(input_nrrd, mask, radius=(7,7,0))
        
        time_end_conversion = time.time()
        time_elapsed_conversion = time_end_conversion - time_start_conversion
        
        if eval:
            time_start_rigid_GT = time.time()
            
            GT_dir = self.config.groundtruth_path        
            GT_nrrd = os.path.join(self.config.output_path,'GT.nrrd')
            if not os.path.isfile(GT_nrrd):
                print("Converting rCT dicom to NRRD...")
                pre_process_tools.convert_dicom_to_nifti(GT_dir, GT_nrrd)
                
            GT_nrrd_resampled = os.path.join(self.config.output_path,'GT_resampled.nrrd')
            pre_process_tools.resample512(GT_nrrd, GT_nrrd_resampled,(1,1,1))                
            self.remove_tb_slices(topslice, bottomslice, img_type='GT')
                
            print("Registering input to ground truth...")
            input_registered = os.path.join(self.config.output_path,'input_registered.nrrd')
            if not os.path.isfile(input_registered):
                pre_process_tools.register(GT_nrrd_resampled, input_nrrd, self.registration_parameters, input_registered)
            else:
                print('Registered image already exists.')
            
            mask = os.path.join(self.config.output_path,'mask.nrrd')
            print("Segmenting Mask...")
            pre_process_tools.segment(input_registered, mask, radius=(7,7,0))
            
            time_end_rigid = time.time()
            time_elapsed_rigid = time_end_rigid - time_start_rigid_GT
                
            # DIR
            if DIR:
                time_start_DIR = time.time()
                
                fixed_image = itk.imread(fr"{GT_nrrd_resampled}",itk.F)
                moving_image = itk.imread(fr"{input_registered}",itk.F)
                input_registered = os.path.join(self.config.output_path,f'{self.img_type}_registered_DIR.nrrd')
                if not os.path.isfile(input_registered):
                    parameter_object = itk.ParameterObject.New()
                    parameter_object.AddParameterFile('/home/galaponav/art/scripts/PhD/generate_sct_paper2/DIR_pelvis.txt')
                    print(parameter_object)

                    output_img, output_transform_parameters = itk.elastix_registration_method(
                        fixed_image, moving_image,
                        parameter_object=parameter_object,
                        log_to_console=True)
                    
                    itk.imwrite(output_img,input_registered,True)
                
                pre_process_tools.segment(input_registered, mask, radius=(7,7,0))
                
                time_end_DIR = time.time()
                time_elapsed_DIR = time_end_DIR - time_start_DIR
            
            GT_masked = os.path.join(self.config.output_path,'GT_masked.nrrd')
            if not os.path.isfile(GT_masked):
                print("Applying mask to GT...")
                pre_process_tools.mask_ct(GT_nrrd_resampled, mask, GT_masked)
            
            # this is final input, will be revised later
            input_masked = os.path.join(self.config.output_path,f'{self.img_type}_masked.nrrd')
            if not os.path.isfile(input_masked):
                print("Applying mask to input...")
                if self.img_type == 'CBCT':
                    pre_process_tools.mask_ct(input_registered, mask, input_masked)
                else:
                    pre_process_tools.mask_mr(input_registered, mask, input_masked) 
            input_nrrd = input_masked            
              
        if self.img_type == 'MRI':
            input_bcorr = os.path.join(self.config.output_path,'MRI_registered_bcorr.nrrd')
            if not os.path.isfile(input_bcorr):
                time_start_bcorr = time.time()
                print("Bias correction for MRI...")
                pre_process_tools.N4Biascorrection(input_nrrd, input_bcorr, mask)
                time_end_bcorr = time.time()
                time_elapsed_bcorr = time_end_bcorr - time_start_bcorr
                
            time_start_histmatch = time.time()
            
            input_masked = os.path.join(self.config.output_path,f'{self.img_type}_masked.nrrd')
            input_histmatched = os.path.join(self.config.output_path,'MRI_registered_histmatched.nrrd')
            #if not os.path.isfile(input_histmatched):
            print("Histogram matching for MRI...")
            pre_process_tools.histogram_matching(input_bcorr, self.reference_MR, input_masked)
            # input_masked used as filename for input_histmatched - will be revised later
            
            time_end_histmatch = time.time()
            time_elapsed_histmatch = time_end_histmatch - time_start_histmatch
            
            
            time_text = os.path.join(os.path.dirname(__file__), 'elapsed_time_preprocessing_steps.txt')
            with open(time_text, 'a') as file:
                file.write(f"{self.config.fname} : {str(time_elapsed_conversion)} : {str(time_elapsed_rigid)} : {str(time_elapsed_bcorr)} : {str(time_elapsed_histmatch)} : {str(time_elapsed_DIR)}\n")            
                
        print("Preprocessing completed.")
            
    def create_dataset(self):
        print("Creating dataset...")
        dataset = ImageDataset(self.config, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,num_workers=0)
        print('Total slices: ',len(dataloader))
        return dataloader
    
    def create_train_dataset(self):
        if self.config.model_type == 'DCNN':
            train_dataset = ImageDataset_train(self.config, train_data=True, normalize=False, augmentation=True)
            val_dataset = ImageDataset_train(self.config, train_data=False, normalize=False, augmentation=False)
        else:
            train_dataset = ImageDataset_train(self.config, train_data=True, normalize=True, augmentation=True)
            val_dataset = ImageDataset_train(self.config, train_data=False, normalize=True, augmentation=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True,num_workers=0)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True,num_workers=0)
        print('Total Train slices: ',len(train_dataloader))
        print('Total Val slices: ',len(val_dataloader))
        return train_dataloader, val_dataloader
    
    def remove_tb_slices(self, numslice_top, numslice_bottom, img_type='CBCT'):
        if img_type == 'CBCT':
            img_nrrd = os.path.join(self.config.output_path,'CBCT.nrrd')
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_nrrd))
        elif img_type == 'GT':
            img_nrrd = os.path.join(self.config.output_path,'GT_resampled.nrrd')
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_nrrd))
        elif img_type == 'MRI':
            img_nrrd = os.path.join(self.config.output_path,'MRI.nrrd')
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_nrrd))
        
        print(f"Removing top {numslice_top} and bottom {numslice_bottom} slices for {img_type}...")
        for slice in range(img_array.shape[0]):
            for topslice in range(slice, slice+numslice_top):
                if img_type == 'CBCT':
                    img_array[topslice] = -1000
                elif img_type == 'MRI':
                    img_array[topslice] = 0
            break

        for slice in range(img_array.shape[0]-1, 0, -1):
            for bottomslice in range(slice, slice-numslice_bottom, -1):
                if img_type == 'CBCT':
                    img_array[bottomslice] = -1000
                elif img_type == 'MRI':
                    img_array[bottomslice] = 0
            break
            
        img_fixed = sitk.GetImageFromArray(img_array)
        img_fixed.CopyInformation(sitk.ReadImage(img_nrrd))
        
        if img_type == 'CBCT':
            sitk.WriteImage(img_fixed, os.path.join(self.config.output_path,'CBCT.nrrd'), True)
        elif img_type == 'rCT':
            sitk.WriteImage(img_fixed, os.path.join(self.config.output_path,'GT_resampled.nrrd'), True)
        elif img_type == 'MRI':
            sitk.WriteImage(img_fixed, os.path.join(self.config.output_path,'MRI_resampled.nrrd'), True)
        
        # generate new mask
        mask_fixed = os.path.join(self.config.output_path,'mask.nrrd')
        pre_process_tools.segment(os.path.join(self.config.output_path,f'{self.img_type}.nrrd'), mask_fixed, radius=(7,7,0))
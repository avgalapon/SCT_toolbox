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
import itk
import SimpleITK as sitk
import pre_process_tools
import torch.utils.data
from ImageDataset import ImageDataset, ImageDataset_train

class PrepareDataset:
    def __init__(self, config, reference_MR=None):
        print("Preparing images...")
        self.config = config
        self.reference_MR = reference_MR
        self.current_dir = os.path.dirname(__file__)
        self.img_type = config.img_type

        # Determine the parameter file based on image type
        param_file = 'parameters_CBCT_brain.txt' if self.img_type == 'CBCT' else 'parameters_MR.txt'
        self.registration_parameters = pre_process_tools.read_parameter_map(
            os.path.join(self.current_dir, 'registration_parameters', param_file)
        )

    def run_sitk(self, topslice, bottomslice, DIR=False, eval=True):
        input_nrrd = self._convert_dicom_to_nifti(self.config.input_path, self.img_type)
        mask = self._segment_image(input_nrrd)
        input_masked = self._apply_mask(input_nrrd, mask, self.img_type)
        
        if eval:
            GT_nrrd = self._convert_dicom_to_nifti(self.config.groundtruth_path, 'GT')
            GT_resampled = self._resample_image(GT_nrrd, 'GT')
            self.remove_tb_slices(topslice, bottomslice, img_type='GT')
            input_registered = self._register_images(GT_resampled, input_nrrd)

            if DIR:
                input_registered = self._deformable_registration(GT_resampled, input_registered)

            mask = self._segment_image(input_registered)
            GT_masked = self._apply_mask(GT_resampled, mask, 'GT')
            input_masked = self._apply_mask(input_registered, mask, self.img_type)

        if self.img_type == 'MRI':
            input_nrrd = self._apply_bias_correction(input_masked, mask)
            self._histogram_matching(input_nrrd)

        print("Preprocessing completed.")

    def create_dataloader(self, dataset, batch_size=1, shuffle=False):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

    def create_dataset(self):
        print("Creating dataset...")
        dataset = ImageDataset(self.config, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,num_workers=0)
        print('Total slices: ',len(dataloader))
        return dataloader

    def create_train_dataset(self):
        normalize = (self.config.model_type != 'DCNN')
        train_dataset = ImageDataset_train(self.config, train_data=True, normalize=normalize, augmentation=True)
        val_dataset = ImageDataset_train(self.config, train_data=False, normalize=normalize, augmentation=False)
        train_dataloader = self.create_dataloader(train_dataset, shuffle=True)
        val_dataloader = self.create_dataloader(val_dataset, shuffle=True)
        print('Total Train slices:', len(train_dataloader))
        print('Total Val slices:', len(val_dataloader))
        return train_dataloader, val_dataloader

    def remove_tb_slices(self, numslice_top, numslice_bottom, img_type='CBCT'):
        img_path = os.path.join(self.config.output_path, f'{img_type}.nrrd')
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        print(f"Removing top {numslice_top} and bottom {numslice_bottom} slices for {img_type}...")

        img_array[:numslice_top] = -1000 if img_type == 'CBCT' else 0
        img_array[-numslice_bottom:] = -1000 if img_type == 'CBCT' else 0

        img_fixed = sitk.GetImageFromArray(img_array)
        img_fixed.CopyInformation(sitk.ReadImage(img_path))
        sitk.WriteImage(img_fixed, img_path, True)

        # Update the mask after slice removal
        self._segment_image(img_path)

    # Helper Methods
    def _convert_dicom_to_nifti(self, input_dir, img_type):
        output_nrrd = os.path.join(self.config.output_path, f"{img_type}.nrrd")
        if not os.path.isfile(output_nrrd):
            print(f"Converting {img_type} DICOM to NRRD...")
            pre_process_tools.convert_dicom_to_nifti(input_dir, output_nrrd)
        return output_nrrd

    def _segment_image(self, input_nrrd, mask_name='mask.nrrd'):
        mask = os.path.join(self.config.output_path, mask_name)
        pre_process_tools.segment(input_nrrd, mask, radius=(7,7,0))
        return mask

    def _resample_image(self, input_nrrd, img_type):
        resampled_path = os.path.join(self.config.output_path, f"{img_type}_resampled.nrrd")
        if not os.path.isfile(resampled_path):
            print(f"Resampling {img_type} to 512...")
            pre_process_tools.resample512(input_nrrd, resampled_path, (1,1,1))
        return resampled_path

    def _register_images(self, fixed_img, moving_img):
        registered_path = os.path.join(self.config.output_path, 'input_registered.nrrd')
        if not os.path.isfile(registered_path):
            print("Registering input to ground truth...")
            pre_process_tools.register(fixed_img, moving_img, self.registration_parameters, registered_path)
        else:
            print("Registered image already exists.")
        return registered_path

    def _deformable_registration(self, fixed_image, moving_image):
        output_path = os.path.join(self.config.output_path, f'{self.config.img_type}_registered_DIR.nrrd')
        if not os.path.isfile(output_path):
            print("Performing deformable registration...")
            parameter_object = itk.ParameterObject.New()
            parameter_object.AddParameterFile(os.path.join(self.current_dir, 'registration_parameters/DIR_parameters.txt'))
            print(parameter_object)
            output_img, _ = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=True
            )
            itk.imwrite(output_img, output_path, True)
        else:
            print("Deformable registration already exists.")
        return output_path

    def _apply_mask(self, input_nrrd, mask, img_type):
        masked_output = os.path.join(self.config.output_path, f'{img_type}_masked.nrrd')
        if not os.path.isfile(masked_output):
            print(f"Applying mask to {img_type}...")
            if img_type == 'CBCT':
                pre_process_tools.mask_ct(input_nrrd, mask, masked_output)
            else:
                pre_process_tools.mask_mr(input_nrrd, mask, masked_output)
        return masked_output

    def _apply_bias_correction(self, input_nrrd, mask):
        corrected_nrrd = os.path.join(self.config.output_path, 'MRI_registered_bcorr.nrrd')
        if not os.path.isfile(corrected_nrrd):
            print("Applying N4 Bias Correction...")
            pre_process_tools.N4Biascorrection(input_nrrd, corrected_nrrd, mask)
        return corrected_nrrd

    def _histogram_matching(self, input_nrrd):
        histmatched_nrrd = os.path.join(self.config.output_path, 'MRI_registered_histmatched.nrrd')
        if not os.path.isfile(histmatched_nrrd):
            print("Histogram matching for MRI...")
            pre_process_tools.histogram_matching(input_nrrd, self.reference_MR, histmatched_nrrd)

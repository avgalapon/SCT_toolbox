import time, sys
import os
from copy import deepcopy
from random import shuffle, randint
import random
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from matplotlib import pyplot as plt

class ImageDataset(Dataset):

    def __init__(self, config, normalize=False):

        self.normalize = normalize
        self.imagetype = config.img_type
        self.dataset_path = config.temp_preprocessed_path
        self.idx_to_files = self.read_file_paths(self.dataset_path)

        self.CT_min = -1024.
        self.CT_max = 3000.
        self.MR_min = 0.
        self.MR_max = 1000.

    def __getitem__(self, index):       
        train_case_path = self.idx_to_files[index]

        slice_X = np.load(train_case_path)
        slice_mask = np.load(train_case_path.replace('X.npy','mask.npy'))

        if self.normalize:
            if self.imagetype == 'cbct':
                slice_X = torch.from_numpy(self.normalizeCT(slice_X)).unsqueeze(0)
            elif self.imagetype == 'mri':
                slice_X = torch.from_numpy(self.normalizeMR(slice_X)).unsqueeze(0)
        else:
            slice_X = torch.from_numpy(slice_X).unsqueeze(0)
        
        slice_mask = torch.from_numpy(slice_mask).unsqueeze(0)

        return {"A": slice_X, "mask": slice_mask}

    def __len__(self):
        return len(self.idx_to_files)

    def read_file_paths(self,datasetPath):
        paths_dir = [os.path.join(datasetPath,p) for p in os.listdir(datasetPath) if 'X.npy' in p]
        return sorted(paths_dir)

    def normalizeCT(self,slice):
        return (slice-self.CT_min)/(self.CT_max-self.CT_min)

    def normalizeMR(self,slice):
        return (slice-self.MR_min)/(self.MR_max-self.MR_min)
    
    
    
# --------------------------------------------------------------------------------------------#
    
class ImageDataset_train(Dataset):
    
    def __init__(self, config, train_data, normalize=False, augmentation=True):

        self.normalize = normalize
        self.augmentation = augmentation
        self.imagetype = config.img_type
        self.modeltype = config.model_type
        self.training_data = train_data

        self.CT_min = -1024.
        self.CT_max = 3000.
        self.MR_min = 0.
        self.MR_max = 1000.
        
        if self.training_data:
            self.dataset_path = os.path.join(config.temp_preprocessed_path, 'train')
        else:
            self.dataset_path = os.path.join(config.temp_preprocessed_path, 'val')
            
        self.idx_to_files = self.read_file_paths(self.dataset_path)

    def __getitem__(self, index):       
        train_case_path = self.idx_to_files[index]

        slice_X = np.load(train_case_path)
        slice_Y = np.load(train_case_path.replace('X.npy','Y.npy'))
        slice_mask = np.load(train_case_path.replace('X.npy','mask.npy'))

        if self.augmentation:
            aug_type = randint(-1,6)
            aug_mag = randint(0,5)
            slice_Y = self.augment_image(slice_Y,aug_type, aug_mag, img_type='CT' )
            slice_mask = self.augment_image(slice_mask,aug_type, aug_mag, img_type='VOI')
            
            if self.imagetype == 'CBCT':
                slice_X = self.augment_image(slice_X,aug_type, aug_mag, img_type='CT' )
            elif self.imagetype == 'MRI':
                slice_X = self.augment_image(slice_X,aug_type, aug_mag, img_type='MRI' )

        if self.normalize:
            if self.imagetype == 'CBCT':
                slice_X = torch.from_numpy(self.normalizeCT(slice_X)).unsqueeze(0)
            elif self.imagetype == 'MRI':
                slice_X = torch.from_numpy(self.normalizeMR(slice_X)).unsqueeze(0)
            slice_Y = torch.from_numpy(self.normalizeCT(slice_Y)).unsqueeze(0)
        else:
            slice_X = torch.from_numpy(slice_X).unsqueeze(0)
            slice_Y = torch.from_numpy(slice_Y).unsqueeze(0)
        slice_mask = torch.from_numpy(slice_mask).unsqueeze(0)

        return {"A": slice_X, "B": slice_Y, "mask": slice_mask}

    def __len__(self):
        return len(self.idx_to_files)

    def read_file_paths(self,datasetPath):
        paths_dir = [os.path.join(datasetPath,p) for p in os.listdir(datasetPath) if 'X.npy' in p]
        return sorted(paths_dir)

    def normalizeCT(self,slice):
        if self.modeltype == 'DCNN':
            normCT = slice
        else:
            # normCT = 2* ((slice-self.CT_min)/(self.CT_max-self.CT_min)) - 1
            normCT = (slice - self.CT_min) / (self.CT_max - self.CT_min)
        return normCT

    def normalizeMR(self,slice):
        if self.modeltype == 'DCNN':
            normMR = slice
        else:
            # normMR = 2* ((slice-self.MR_min)/(self.MR_max-self.MR_min)) - 1
            normMR = (slice - self.MR_min) / (self.MR_max - self.MR_min)
        return normMR
    
    def augment_image(self, img, augmentation_type, augmentation_magnitude, img_type):
        """
        This function augments the image. It can be unchanged, mirrored or traslated.
        """
        # Create empty image according to image type
        if img_type == 'MRI' or img_type == 'VOI':
            augmented_img = np.zeros_like(img)
        elif img_type == 'CT':
            augmented_img = np.ones_like(img)*-1000

        # Modify image according to the augmentation type
        if augmentation_type == -1:
            augmented_img[:,:] = img[:,:]
        elif augmentation_type == 0:
            augmented_img[int(augmentation_magnitude+1):,:] = img[:-int(augmentation_magnitude+1),:]
        elif augmentation_type == 1:
            augmented_img[:-int(augmentation_magnitude+1),:] = img[int(augmentation_magnitude+1):,:]
        elif augmentation_type == 2:
            augmented_img[:,int(augmentation_magnitude+1):] = img[:,:-int(augmentation_magnitude+1)]
        elif augmentation_type == 3:
            augmented_img[:,:-int(augmentation_magnitude+1)] = img[:,int(augmentation_magnitude+1):]
        elif augmentation_type == 4:
            augmented_img[:,:] = img[:,::-1]
        elif augmentation_type == 5:
            augmented_img[:,:] = img[::-1,:]
        elif augmentation_type == 6:
            augmented_img[:,:] = img[::-1,::-1]

        return augmented_img
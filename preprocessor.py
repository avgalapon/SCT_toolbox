import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import shutil


class Preprocessor:
    def __init__(self,config):
        print("Preprocessing...")
        self.save_path = config.temp_preprocessed_path
        self.output_path = config.output_path
        self.img_type = config.img_type
        self.X = config.X
        self.VOI = config.VOI
        self.orientation = config.orientation
        
        if os.path.isdir(self.save_path):
            print("Deleting existing preprocessed data...")
            shutil.rmtree(self.save_path)
        
        os.mkdir(self.save_path)

    def read_image(self, img_fn, orientation):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_fn)).astype(np.float32)
        img = self.rotate_image(img, orientation)
        if self.img_type == 'MRI':
            img, offsets = self.reshape_image(img, background_value=0)
        return img
    
    def reshape_image(self, img, background_value=0.0):
        """
        This function reshapes the image to a cubic matrix.
        """
        img_shape = img.shape
        # max_image_shape = max(img_shape)
        max_image_shape = 512
        offsets = (int(np.floor(max_image_shape-img_shape[0])/2.0), int(np.floor(max_image_shape-img_shape[1])/2.0), int(np.floor(max_image_shape-img_shape[2])/2.0))

        reshaped_img = np.ones((img_shape[0], max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
        reshaped_img[:,offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:] #,offsets[0]:offsets[0]+img_shape[0]

        return reshaped_img.astype(np.float32), offsets
    
    def rotate_image(self, img, orientation):
        """
        This function swaps axes in order to have the data in SAC format.
        """

        if orientation == "axial": # Sagittal, Axial, Coronal
            return img
        elif orientation == "sagittal": # Axial, Coronal, Sagittal:
            m1 = np.swapaxes(img, 0, 2) # CAS
            m1 = np.rot90(m1, k=1, axes=(1,2))
            return m1
        elif orientation == "coronal":
            m1 = np.swapaxes(img, 0, 1) # CAS
            m1 = np.rot90(m1, k=2, axes=(1,2))
            return m1
        else:
            print("Unknow orientation...Exit")
            exit()
    
    def preprocess(self):
        VOI = self.read_image(os.path.join(self.output_path, self.VOI), self.orientation)
        X = self.read_image(os.path.join(self.output_path, self.X), self.orientation)
        for slice in range(X.shape[0]):
            slice_X = X[slice]
            slice_voi = VOI[slice]
            if slice < 10:
                np.save(os.path.join(self.save_path,f'00{slice}_X'),slice_X)
                np.save(os.path.join(self.save_path,f'00{slice}_mask'),slice_voi)
            elif slice < 100:
                np.save(os.path.join(self.save_path,f'0{slice}_X'),slice_X)
                np.save(os.path.join(self.save_path,f'0{slice}_mask'),slice_voi)
            else:
                np.save(os.path.join(self.save_path,f'{slice}_X'),slice_X)
                np.save(os.path.join(self.save_path,f'{slice}_mask'),slice_voi)
        print("Preprocessing Complete.")

class Preprocessor_train:
    def __init__(self,config):
        print("Preprocessing...")
        self.save_path = config.temp_preprocessed_path
        self.output_path = config.output_path
        self.X = config.X
        self.Y = config.Y
        self.VOI = config.VOI
        self.orientation = config.orientation
        
        if os.path.isdir(self.save_path):
            print("Deleting existing preprocessed data...")
            shutil.rmtree(self.save_path)
        
        os.mkdir(self.save_path)

    def read_image(self, img_fn, orientation):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_fn)).astype(np.float32)
        img = self.rotate_image(img, orientation)
        return img
    
    def rotate_image(self, img, orientation):
        """
        This function swaps axes in order to have the data in SAC format.
        """

        if orientation == "axial": # Sagittal, Axial, Coronal
            return img
        elif orientation == "sagittal": # Axial, Coronal, Sagittal:
            m1 = np.swapaxes(img, 0, 2) # CAS
            m1 = np.rot90(m1, k=1, axes=(1,2))
            return m1
        elif orientation == "coronal":
            m1 = np.swapaxes(img, 0, 1) # CAS
            m1 = np.rot90(m1, k=2, axes=(1,2))
            return m1
        else:
            print("Unknow orientation...Exit")
            exit()
    
    def preprocess_train(self, patient_list, train_data = True):
        if train_data:
            savepath = os.path.join(self.save_path,'train')
        else:
            savepath = os.path.join(self.save_path,'val')
            
        if os.path.isdir(savepath):
            print("Deleting existing preprocessed training/validation data...")
            shutil.rmtree(savepath)
        os.mkdir(savepath)
        
        patient_progressbar = tqdm(patient_list)
        for i,patient in enumerate(patient_progressbar):
            patient_progressbar.set_description(f'Preprocessing image: {patient}')
            case_path = os.path.join(self.output_path,patient)
            VOI = self.read_image(os.path.join(case_path, self.VOI), self.orientation)
            X = self.read_image(os.path.join(case_path, self.X), self.orientation)
            Y = self.read_image(os.path.join(case_path, self.Y), self.orientation)
            
            for slice in range(X.shape[0]):
                if VOI[slice].max() >0: # if training
                    slice_X = X[slice]
                    slice_Y = Y[slice]
                    slice_voi = VOI[slice]
                    if slice < 10:
                        np.save(os.path.join(savepath,f'{patient}_00{slice}_X'),slice_X)
                        np.save(os.path.join(savepath,f'{patient}_00{slice}_Y'),slice_Y)
                        np.save(os.path.join(savepath,f'{patient}_00{slice}_mask'),slice_voi)
                    elif slice < 100:
                        np.save(os.path.join(savepath,f'{patient}_0{slice}_X'),slice_X)
                        np.save(os.path.join(savepath,f'{patient}_0{slice}_Y'),slice_Y)
                        np.save(os.path.join(savepath,f'{patient}_0{slice}_mask'),slice_voi)
                    else:
                        np.save(os.path.join(savepath,f'{patient}_{slice}_X'),slice_X)
                        np.save(os.path.join(savepath,f'{patient}_{slice}_Y'),slice_Y)
                        np.save(os.path.join(savepath,f'{patient}_{slice}_mask'),slice_voi)
        print("Preprocessing Complete.")
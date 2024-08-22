import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


class Preprocessor:
    def __init__(self,lp,save_path):
        self.lp = lp
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    def read_image(self, img_fn, orientation, background_value):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_fn)).astype(np.float32)
        img = self.rotate_image(img, orientation)
        img = self.reshape_image(img, background_value)
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

    def reshape_image(self, img, background_value=0.0):
        """
        This function reshapes the image to a cubic matrix.
        """

        img_shape = img.shape
        # max_image_shape = max(img_shape)
        max_image_shape = 512
        offsets = (int(np.floor(max_image_shape-img_shape[0])/2.0), int(np.floor(max_image_shape-img_shape[1])/2.0), int(np.floor(max_image_shape-img_shape[2])/2.0))

        reshaped_img = np.ones((max_image_shape, max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
        reshaped_img[offsets[0]:offsets[0]+img_shape[0],offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:]

        return reshaped_img.astype(np.float32)

    def preprocess(self,case_list):
        progressbar = tqdm(case_list)
        for i,case in enumerate(progressbar):
            progressbar.set_description(f'Preprocessing image: {case}')
            train_case_path = self.lp["dataPath"] + case + os.sep
            Y= self.read_image(train_case_path+self.lp["Y"], self.lp["orientation"], background_value=-1000) # This is for CT
            VOI = self.read_image(train_case_path+self.lp["VOI"], self.lp["orientation"], background_value=0) # Volume of Interst based on skin mask
            VOI = VOI >0
            X = self.read_image(train_case_path+self.lp['Xchannels'][0], self.lp["orientation"], background_value=0) # what is the additional channel index for

            for slice in range(X.shape[0]):
                if VOI[slice].max() >0:
                    slice_X = X[slice]
                    slice_Y = Y[slice]
                    slice_voi = VOI[slice]
                    np.save(os.path.join(self.save_path,f'{case}_{slice}_X.npy'),slice_X)
                    np.save(os.path.join(self.save_path,f'{case}_{slice}_Y.npy'),slice_Y)
                    np.save(os.path.join(self.save_path,f'{case}_{slice}_mask.npy'),slice_voi)
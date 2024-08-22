#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_dropout.py
@Time    :   2023/01/23 11:25:25
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   DCNN sCT gen with dropout
'''
import os
import time
import shutil
import logging
import argparse
import numpy as np
import SimpleITK as sitk
import concurrent.futures
from config_class import Config
import model_class
import preprocessor as Preprocess
from synthesis_class import Generate_sCT
from evaluation_class import Evaluate_sCT
import prepare_data_nrrd_class as prepare_data_class
from utils import return_to_HU_cycleGAN, return_to_HU_cGAN

# Set up logging
logging.basicConfig(
    filename='test_dropout.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_inference(cfg, num_inf):
    try:
        prepare_data = prepare_data_class.prepare_dataset(cfg, reference_MR='/data/galaponav/dataset/newHN_MR2/p0024/MRI_registered_bcorr_axial.nrrd') 
        model, device = model_class.Model(cfg).initialize_models()
        dataloader = prepare_data.create_dataset()
        sct_gen = Generate_sCT(cfg, model, dataloader, device)
        logger.info(f"Iteration #: {num_inf+1}")
        return sct_gen.inference_loop()
    except Exception as e:
        logger.error(f"Error during inference at iteration {num_inf+1}: {e}")
        raise

def save_generated_sct(cfg, predictions):
    try:
        logger.info("Stacking slices for sCT generation...")
        sct_stack = np.stack(predictions)
        sct_mean, sct_variance = np.mean(sct_stack, axis=0), np.var(sct_stack, axis=0)
        reference_input = sitk.ReadImage(os.path.join(cfg.output_path, cfg.X))
        mask = sitk.ReadImage(os.path.join(cfg.output_path, cfg.VOI))

        sct_img, unc_img = np.squeeze(sct_mean), np.squeeze(sct_variance)

        if cfg.img_type == 'MRI':
            sct_img, unc_img = revert_image(sct_img, reference_input), revert_image(unc_img, reference_input)

        logger.info('Updating image properties...')
        sct_img.CopyInformation(reference_input)
        unc_img.CopyInformation(reference_input)

        sct_img = sitk.Mask(sct_img, mask, -1000, 0)
        unc_img = sitk.Mask(unc_img, mask, 0, 0)

        sct_img = unnorm_sct(cfg, sct_img)
        unc_img = unnorm_sct(cfg, unc_img)

        logger.info('Saving sCT-nrrd...')
        sct_fname = os.path.join(cfg.output_path, f'sCT_{cfg.model_type}_{cfg.fname}.nrrd')
        sitk.WriteImage(sct_img, sct_fname, True)

        if cfg.dropout_enable:
            unc_fname = os.path.join(cfg.output_path, f'epistemic_{cfg.model_type}_{cfg.fname}.nrrd')
            sitk.WriteImage(unc_img, unc_fname, True)
            logger.info(f"Files saved: {sct_fname}, {unc_fname}")
        else:
            logger.info(f"File saved: {sct_fname}")
    except Exception as e:
        logger.error(f"Error saving generated sCT: {e}")
        raise

def revert_image(img, img_ref):
    try:
        img_ref_arr = sitk.GetArrayFromImage(img_ref)
        original_shape = img_ref_arr.shape
        offsets = [(512 - dim) // 2 for dim in original_shape]
        original_img = img[:, offsets[1]:offsets[1] + original_shape[1], offsets[2]:offsets[2] + original_shape[2]]
        return sitk.GetImageFromArray(original_img.astype(np.float32))
    except Exception as e:
        logger.error(f"Error in reverting image: {e}")
        raise

def unnorm_sct(cfg, sct_arr):
    try:
        if cfg.model_type == 'cycleGAN':
            sct_arr = return_to_HU_cycleGAN(sct_arr)
        elif cfg.model_type == 'cGAN':
            sct_arr = return_to_HU_cGAN(sct_arr)
        return sct_arr
    except Exception as e:
        logger.error(f"Error in unnormalizing sCT: {e}")
        raise

def save_data_uncertainty(cfg, uncertainty):
    try:
        logger.info("Stacking slices for uncertainty...")
        variance = np.mean([u**2 for u in uncertainty], axis=0)
        combined_std = np.sqrt(variance)

        unc_img = sitk.GetImageFromArray(np.squeeze(combined_std))
        reference_cbct = sitk.ReadImage(os.path.join(cfg.output_path, cfg.X))
        unc_img.CopyInformation(reference_cbct)
        unc_img = unnorm_sct(cfg, unc_img)

        unc_fname = os.path.join(cfg.output_path, f'aleatoric_{cfg.model_type}_{cfg.fname}.nrrd')
        sitk.WriteImage(unc_img, unc_fname, True)
        logger.info(f"File saved: {unc_fname}")
    except Exception as e:
        logger.error(f"Error saving uncertainty data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate sCT image')
    parser.add_argument("json_parameters_path", help="Path to the test parameters file")
    args = parser.parse_args()

    time_start = time.time()
    logger.info("Starting sCT generation process...")

    try:
        cfg = Config(args.json_parameters_path)

        prepare_data = prepare_data_class.prepare_dataset(cfg, reference_MR='/data/galaponav/dataset/newHN_MR2/p0024/MRI_registered_bcorr_axial.nrrd')
        prepare_data.run_sitk(1, 1, DIR=cfg.DIR, eval=cfg.EVAL)

        Preprocess.Preprocessor(cfg).preprocess()

        with open(os.path.join(os.path.dirname(__file__), 'elapsed_time_prep.txt'), 'a') as file:
            file.write(f"{cfg.fname} : {time.time() - time_start}\n")

        model, device = model_class.Model(cfg).initialize_models()
        num_inference = model_class.Model(cfg).enable_dropout(model)

        time_synthesis_start = time.time()
        predictions, uncertainties = [], []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_inference, cfg, num_inf) for num_inf in range(num_inference)]
            for future in concurrent.futures.as_completed(futures):
                ct_gen, unc_gen = future.result()
                predictions.append(ct_gen)
                uncertainties.append(unc_gen)

        save_generated_sct(cfg, predictions)
        if cfg.model_type == 'cGAN':
            save_data_uncertainty(cfg, uncertainties)

        shutil.rmtree(cfg.temp_preprocessed_path, ignore_errors=True)

        logger.info(f'sCT generation completed! Total time: {time.time() - time_start} seconds, synthesis time: {time.time() - time_synthesis_start} seconds')

        with open(os.path.join(os.path.dirname(__file__), 'elapsed_time.txt'), 'a') as file:
            file.write(f"{cfg.fname} : {time.time() - time_start} : {time.time() - time_synthesis_start}\n")

        if cfg.EVAL:
            EvaluateIQ = Evaluate_sCT(cfg)
            EvaluateIQ.run_IQeval(250)
    except Exception as e:
        logger.error(f"Error during main process: {e}")
        raise

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
import os
import fnmatch
import pre_process_tools as pre

datapath = fr"/data/galaponav/dataset/newHN_MR2/" #DATA_MRI_CT /home/art/MDACC/DATA_MRI_CT
patient_list = os.listdir(datapath)
overview_path = fr"/data/galaponav/output/newHN_MR2/overview"
no_folder = []
orientation = 'axial'

# patient_list = ['6038065'] #ref
print(patient_list)

for patient in sorted(patient_list):
    # try:
        print(fr"Processing patient {patient}")

        ## pCT
        # print(os.listdir(os.path.join(datapath, patient)))
        # ct_list_NC = fnmatch.filter(os.listdir(os.path.join(datapath, patient)), '*CT*')
        # pct_list = fnmatch.filter(os.listdir(os.path.join(datapath, patient,ct_list_NC[0])), '*pCT*')
        # ct_folder = os.path.join(datapath, patient,ct_list_NC[0],pct_list[0])
        
        # pCT_nrrd = os.path.join(datapath,patient,f'pCT_{orientation}.nrrd')
        # if os.path.isfile(pCT_nrrd):
        #     print('--pCT already converted')
        # else:
        #     print('--convert pCT')
        #     pre.convert_dicom_to_nifti(ct_folder, pCT_nrrd)
        
        pCT_resampled = os.path.join(datapath,patient, f'pCT_resampled_{orientation}.nrrd')
        # if os.path.isfile(pCT_resampled):
        #     print('--pCT already resampled!')
        # else:
        #     print('--resample pCT')
        #     pre.resample512b(pCT_nrrd, pCT_resampled, (1,1,1), orientation)

        # ## MRI
        # MR_filter = fnmatch.filter(os.listdir(os.path.join(datapath,patient)), '*MRI*')
        # MR_filter_folder = [item for item in MR_filter if os.path.isdir(os.path.join(datapath,patient,item))]
        # MR_folder = os.path.join(datapath,patient,MR_filter_folder[0])
        # MRI_fn = [item for item in os.listdir(MR_folder) if os.path.isdir(os.path.join(MR_folder, item))][0]
        
        mr_nrrd = os.path.join(datapath,patient,f'MRI_{orientation}.nrrd')
        # if os.path.isfile(mr_nrrd):
        #     print('--MRI already converted')
        # else:
        #     print('--convert MRI')
        #     pre.convert_dicom_to_nifti(os.path.join(datapath,patient,MR_filter_folder[0],MRI_fn), mr_nrrd)

        ## Register MRI to CT
        mr_registered = os.path.join(datapath,patient,f'MRI_registered_{orientation}.nrrd') 
        if os.path.isfile(mr_registered):
            print('--MRI already registered!')
        else:
            print('--register MRI')
            pre.register(pCT_resampled,mr_nrrd,
                        pre.read_parameter_map(r'/home/art/MDACC/MR_MDACC/parameters_MR.txt'),
                        mr_registered)

        ## Find mask MR and CT
        mask_mr = os.path.join(datapath,patient,f'mask_MRI_{orientation}.nrrd')
        if os.path.isfile(mask_mr):
            print('--MRI already segmented!')
        else:
            print('--segment MRI')
            pre.segment3(mr_registered, mask_mr)
        
        ## Correct mask to fit MR FOV
        mask_mr_corrected = os.path.join(datapath,patient,f'mask_MRI_corrected_{orientation}.nrrd')
        if os.path.isfile(mask_mr_corrected):
            print('--mask already corrected!')
        else:
            print('--correct mask')
            pre.correct_mask_mr(mr_nrrd, pCT_resampled, os.path.join(datapath,patient,f'MRI_registered_{orientation}_parameters.txt'), mask_mr, mask_mr_corrected)

        # Apply mask to CT
        pCT_masked = os.path.join(datapath,patient,f'pCT_resampled_masked_{orientation}.nrrd')
        if os.path.isfile(pCT_masked):
            print('--mask already applied to pCT')
        else:
            print('--applying mask to CT')
            pre.mask_ct(pCT_resampled, mask_mr_corrected, pCT_masked)
            
        # N4 Bias Correction
        folders_MR = os.listdir(os.path.join(datapath,patient))
        # MR_fn = fnmatch.filter(folders_MR,'*MRI_registered.*')[0]
        
        mr_bcorr = os.path.join(datapath,patient,f'MRI_registered_bcorr_{orientation}.nrrd')
        if os.path.isfile(mr_bcorr):
                print('--Bias already corrected')
        else:
            print('--applying N4 Bias Correction to MR')
            pre.N4Biascorrection(mr_registered, mr_bcorr, mask_mr_corrected)
            
        # Histogram Matching
        folders_MR = os.listdir(os.path.join(datapath,patient))
        reference_hist = fr'{datapath}/p0024/MRI_registered_bcorr_{orientation}.nrrd'

        mr_histmatched = os.path.join(datapath,patient,f'MRI_registered_bcorr_histmatched_{orientation}.nrrd')
        # if os.path.isfile(mr_histmatched):
            # print('--Histogram already matched')
        # else:
            # print('--performing histogram matching to MR using reference')
            # matched = pre.histogram_matching(mr_bcorr,reference_hist,mr_histmatched)
            
        print('--performing histogram matching to MR using reference')
        matched = pre.histogram_matching(mr_bcorr,reference_hist,mr_histmatched)
            
        ## crop MR and CT without applying any mask
        # if os.path.isfile(os.path.join(datapath, patient, 'MRI_registered_bcorr_histmatched_cropped.nrrd')):
        #     print('--MR already cropped!')
        # else:
        #     print('--crop MR')
        #     pre.crop(os.path.join(datapath, patient, 'MRI_cropped_bcorr_histmatched.nrrd'),
        #                 os.path.join(datapath, patient, 'mask_MRI_corrected.nrrd'),
        #                 os.path.join(datapath, patient, 'MRI_registered_bcorr_histmatched_cropped.nrrd'))

        # if os.path.isfile(os.path.join(datapath, patient, 'CT_resampled_masked_cropped.nrrd')):
        #     print('--CT already cropped!')
        # else:
        #     print('--crop CT')
        #     pre.crop(os.path.join(datapath, patient, 'pCT_resampled_masked.nrrd'),
        #                 os.path.join(datapath, patient, 'mask_MRI_corrected.nrrd'),
        #                 os.path.join(datapath, patient, 'CT_resampled_masked_cropped.nrrd'))

        # if os.path.isfile(os.path.join(datapath, patient, 'mask_cropped.nrrd')):
        #     print('--mask already cropped!')
        # else:
        #     print('--crop mask')
        #     pre.crop(os.path.join(datapath, patient, 'mask_MRI_corrected.nrrd'),
        #                 os.path.join(datapath, patient, 'mask_MRI_corrected.nrrd'),
        #                 os.path.join(datapath, patient, 'mask_cropped.nrrd'))

        ## generate overview image
        # if os.path.isfile(os.path.join(path, patient, 'overview_MRI.png')):
        # if os.path.isfile(os.path.join(overview_path, f"overview_MRI_{patient}.png")):
        #     print('--overview already generated!')
        # else:
        #     print('--generating overview')
        #     pre.generate_overview(mr_histmatched, pCT_masked, mask_mr_corrected,
        #                             os.path.join(overview_path, f"overview_MRI_{patient}.png"), title=patient)
        pre.generate_overview(mr_histmatched, pCT_masked, mask_mr_corrected,
                                    os.path.join(overview_path, f"overview_MRI_{patient}.png"), title=patient)
#     except:
#         print(f'Patient {patient} has no C folder')
#         no_folder.append(patient)
# print(no_folder)
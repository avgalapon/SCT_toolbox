import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from operator import mul

def convert_dicom_to_nifti(input, output):
    print("Converting DICOM to NRRD/NIFTI...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output, True)

def resample(input, output, spacing):
    image = sitk.ReadImage(input)
    space = image.GetSpacing()
    size = image.GetSize()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    new_space = spacing
    new_size = tuple([int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(size, space, new_space)])
    image_resampled = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear, origin, new_space, direction,
                                    -1000.0, sitk.sitkInt16)
    sitk.WriteImage(image_resampled, output,True)
    
def resample512(input, output, spacing):
    image = sitk.ReadImage(input)
    space = image.GetSpacing()
    size = image.GetSize()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    new_space = spacing
    new_size_xy = (512, 512)
    z_size = int(round(size[2] * space[2] / new_space[2]))
    new_size = (new_size_xy[0], new_size_xy[1], z_size)
    image_resampled = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear, origin, new_space, direction,
                                    -1000.0, sitk.sitkInt16)
    sitk.WriteImage(image_resampled, output,True)

def read_parameter_map(parameter_fn):
    return sitk.ReadParameterFile(parameter_fn)

def register(fixed, moving, parameter, output):
    # Load images
    fixed_image = sitk.ReadImage(fixed)
    moving_image = sitk.ReadImage(moving)

    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(moving_image)  # due to FOV differences CT first registered to MR an inverted in the end
    elastixImageFilter.SetMovingImage(fixed_image)
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.Execute()

    # convert to itk transform format
    transform = elastixImageFilter.GetTransformParameterMap(0)
    x = transform.values()
    center = np.array((x[0])).astype(np.float64)
    rigid = np.array((x[22])).astype(np.float64)
    transform_itk = sitk.Euler3DTransform()
    transform_itk.SetParameters(rigid)
    transform_itk.SetCenter(center)
    transform_itk.SetComputeZYX(False)

    # save itk transform to correct MR mask later
    output = str(output)
    # transform_itk.WriteTransform(str(output.split('.')[:-2][0]) + '_parameters.txt')
    transform_itk.WriteTransform(str(output.split('.')[0]) + '_parameters.txt')
    #transform_itk.WriteTransform(str('registration_parameters.txt'))

    ##invert transform to get MR registered to CT
    inverse = transform_itk.GetInverse()

    ## check if moving image is an mr or cbct
    min_moving = np.amin(sitk.GetArrayFromImage(moving_image))
    if min_moving <-500:
        background = -1000
    else:
        background = 0

    ##transform MR image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    resample.SetTransform(inverse)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(background)
    output_image = resample.Execute(moving_image)

    # write output image
    sitk.WriteImage(output_image, output, True)

def clean_border(input_image, output_image):
    im = (sitk.ReadImage(input_image))
    im_np = sitk.GetArrayFromImage(im)
    im_np[im_np >= 3500] = 0
    im2 = sitk.GetImageFromArray(im_np)
    im2.CopyInformation(im)
    sitk.WriteImage(im2, output_image,True)

def segment(input_image, output_mask, radius=(12, 12, 12)):
    print("Segmenting Mask...")
    image = sitk.InvertIntensity(sitk.Cast(sitk.ReadImage(input_image),sitk.sitkFloat32))
    mask = sitk.OtsuThreshold(image)
    dil_mask = sitk.BinaryDilate(mask, (1, 1, 0))
    component_image = sitk.ConnectedComponent(dil_mask)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    mask_closed = sitk.BinaryMorphologicalClosing(largest_component_binary_image, radius)
    # dilated_mask = sitk.BinaryDilate(mask_closed, (10, 10, 0))
    dilated_mask = sitk.BinaryDilate(mask_closed, (1, 1, 0))
    filled_mask = sitk.BinaryFillhole(dilated_mask)
    sitk.WriteImage(filled_mask, output_mask,True)
    
def segment2(input_image, output_mask=None, radius=(12, 12, 12),return_sitk=False):
    image = sitk.InvertIntensity(sitk.Cast(sitk.ReadImage(input_image),sitk.sitkFloat32))
    mask = sitk.OtsuThreshold(image)
    # dil_mask = sitk.BinaryDilate(mask, (5, 5, 0))
    component_image = sitk.ConnectedComponent(mask)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    mask_closed = sitk.BinaryMorphologicalClosing(largest_component_binary_image, (12, 12, 12))
    dilated_mask = sitk.BinaryDilate(mask_closed, (5, 5, 0))
    filled_mask = sitk.BinaryFillhole(dilated_mask)
    if return_sitk:
        return filled_mask
    else:
        sitk.WriteImage(filled_mask, output_mask, True)
    
def segment_mask_thresholding(cbct, lower_thresh, upper_thresh, radius=(12, 12, 12)):
    cbct_img = sitk.Cast(sitk.ReadImage(cbct), sitk.sitkFloat32)

    mask = np.zeros_like(sitk.GetArrayFromImage(cbct_img))

    for z in range(cbct_img.GetDepth()):
        slice_img = cbct_img[:,:,z]
        
        if sitk.GetArrayViewFromImage(slice_img).max() == 0:
            continue
        
        smoothed_slice = sitk.SmoothingRecursiveGaussian(slice_img, 1.0)
        slice_mask = sitk.BinaryThreshold(smoothed_slice, lowerThreshold=lower_thresh, upperThreshold=upper_thresh, insideValue=1, outsideValue=0)
        slice_mask = sitk.BinaryMorphologicalClosing(slice_mask, radius)
        slice_mask = sitk.ConnectedComponent(slice_mask)
        slice_mask = sitk.RelabelComponent(slice_mask, sortByObjectSize=True)
        slice_mask = slice_mask == 1
        # slice_mask = sitk.BinaryErode(slice_mask, [8,8])
        slice_mask = sitk.BinaryDilate(slice_mask, [1,1])
        slice_mask = sitk.BinaryFillhole(slice_mask)
        mask[z] = sitk.GetArrayFromImage(slice_mask)

    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(cbct_img)

    return mask

def correct_mask_mr(mr,ct,transform,mask,mask_corrected):
    # load inputs
    mr_im = sitk.ReadImage(mr)
    ct_im = sitk.ReadImage(ct)
    tf = sitk.ReadTransform(transform)
    mask_im = sitk.ReadImage(mask)
    
    # create mask of original MR FOV
    mr_im_np = sitk.GetArrayFromImage(mr_im)
    mr_im_np[mr_im_np>=-2000]=1
    fov=sitk.GetImageFromArray(mr_im_np)
    fov.CopyInformation(mr_im)

    # transform mask to registered mr
    tf = tf.GetInverse()
    default_value=0
    interpolator=sitk.sitkNearestNeighbor
    fov_reg = sitk.Resample(fov,ct_im,tf,interpolator,default_value)
    
    # correct MR mask with fov_reg box
    fov_np = sitk.GetArrayFromImage(fov_reg)
    mask_np = sitk.GetArrayFromImage(mask_im)
    mask_corrected_np=mask_np*fov_np

    #save corrected mask
    mask_corrected_im = sitk.GetImageFromArray(mask_corrected_np)
    mask_corrected_im.CopyInformation(mask_im)
    sitk.WriteImage(mask_corrected_im, mask_corrected,True)

def mask_ct(input_image, input_mask, output_image):
    image = sitk.ReadImage(input_image)
    mask = sitk.Cast(sitk.ReadImage(input_mask), sitk.sitkUInt8)
    masked_image = sitk.Mask(image, mask, -1000, 0)
    sitk.WriteImage(masked_image, output_image,True)

def mask_mr(input_image, input_mask, output_image):
    image = sitk.ReadImage(input_image)
    mask = sitk.ReadImage(input_mask)
    masked_image = sitk.Mask(image, mask, 0, 0)
    sitk.WriteImage(masked_image, output_image,True)

def crop(input_image, mask_for_crop, output_image):
    image = sitk.ReadImage(input_image)
    mask = sitk.ReadImage(mask_for_crop)
    mask_np = sitk.GetArrayFromImage(mask)
    idx_nz = np.nonzero(mask_np)
    IP = [np.min(idx_nz[0]) , np.max(idx_nz[0]) ]
    border=10
    if  np.min(idx_nz[1])<border:
        AP = [np.min(idx_nz[1]) - np.min(idx_nz[1]), np.max(idx_nz[1]) + 10]
    else:
        AP = [np.min(idx_nz[1]) - border, np.max(idx_nz[1]) + border]
    if np.min(idx_nz[2]) < border:
        LR = [np.min(idx_nz[2]) - np.min(idx_nz[2]), np.max(idx_nz[2]) + 10]
    else:
        LR = [np.min(idx_nz[2]) - border, np.max(idx_nz[2]) + border]
    cropped_image = image[LR[0]:LR[1], AP[0]:AP[1], IP[0]:IP[1]]
    sitk.WriteImage(cropped_image, output_image,True)


def N4Biascorrection(input_image, output_image, mask):
    print("Applying bias correction for MRI...")
    input = sitk.ReadImage(input_image, sitk.sitkFloat32)
    image = input
    maskimage = sitk.ReadImage(mask, sitk.sitkUInt8)
    correctedImage = sitk.N4BiasFieldCorrection(image, maskimage)
    correctedImage.CopyInformation(input)
    sitk.WriteImage(correctedImage, output_image,True)
    return

def histogram_matching(mov_scan, ref_scan, output,
                       histogram_levels=2048,
                       match_points=100,
                       set_th_mean=True):
    """
    Histogram matching following the method developed on
    Nyul et al 2001 (ITK implementation)
    inputs:
    - mov_scan: np.array containing the image to normalize
    - ref_scan np.array containing the reference image
    - histogram levels
    - number of matched points
    - Threshold Mean setting
    outputs:
    - histogram matched image
    """

    # convert np arrays into itk image objects
    ref = sitk.ReadImage(ref_scan)
    mov = sitk.ReadImage(mov_scan)
    
    # ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    # mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    # perform histogram matching
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref.GetPixelID())

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(set_th_mean)
    matched_vol = matcher.Execute(mov, ref)
    sitk.WriteImage(matched_vol,output,True)

    return sitk.GetArrayFromImage(matched_vol)

def generate_overview(input_path,ref_path,mask_path,output_path,title=''):
    #load images as np arrrays
    im=sitk.ReadImage(ref_path)
    ref_img = sitk.GetArrayFromImage(im)
    input_img = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
    mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    
    # attempt of 'normalizing' images for difference calculations
    if np.max(ref_img)>3000:
        max_ref = 3000
    else:
        max_ref = np.max(ref_img)
    
    if np.max(input_img)>2000:
        max_in = 2000
    else:
        max_in = np.max(input_img)

    input_norm = (input_img+np.abs(np.min(input_img)))/(max_in+np.abs(np.min(input_img)))
    ref_norm = (ref_img+np.abs(np.min(ref_img)))/(max_ref+np.abs(np.min(ref_img)))
    diff = input_norm - ref_norm

    # select central slices
    im_shape = np.shape(ref_img)

    #aspect ratio for plots
    spacing=list(im.GetSpacing())
    spacing.reverse() #SimpleITK to numpy conversion
    asp_ax = spacing[1]*spacing[2]
    asp_sag = spacing[0]*spacing[1]
    asp_cor = spacing[0]*spacing[2]

    #window/level for CBCT/MR (called input)
    if np.min(input_img)<-500: #CBCT
        if spacing[0]==1:   #brain
            w_i = 2500
            l_i = 250
        else:               #pelvis
            w_i = 1200
            l_i = -400
    else: #MR
        if spacing[0]==1:   #brain
            w_i = 600
            l_i = 280
        else:               #pelvis
            w_i = 600
            l_i = 280
    
    #window/level for CT (called ref)
    w_r = 2500
    l_r = 200

    # titles for subplots
    titles = [  os.path.basename(os.path.normpath(input_path)),
                os.path.basename(os.path.normpath(ref_path)),
                os.path.basename(os.path.normpath(mask_path)),
                'Difference'
                ]

    # make subplots axial, sagittal and coronal view
    fig, ax = plt.subplots(3,4,figsize=(14,10))

    fig.suptitle(title, fontsize=18,y=1.01)

    ax[0][0].imshow(input_img[int(im_shape[0]/2),:,::-1],cmap='gray',aspect=asp_ax,vmin=l_i-w_i/2,vmax=l_i+w_i/2)
    ax[0][1].imshow(ref_img[int(im_shape[0]/2),:,::-1],cmap='gray',aspect=asp_ax,vmin=l_r-w_r/2,vmax=l_r+w_r/2)
    ax[0][2].imshow(mask_img[int(im_shape[0]/2),:,::-1],cmap='gray',aspect=asp_ax)
    ax[0][3].imshow(diff[int(im_shape[0]/2),:,::-1],cmap='RdBu',aspect=asp_ax,vmin=-0.3,vmax=0.3)

    for j in range(4):
        ax[1][j].set_xticklabels([])
        ax[1][j].set_yticklabels([])

    ax[1][0].imshow(input_img[::-1,:,int(im_shape[2]/2)],cmap='gray',aspect=asp_sag,vmin=l_i-w_i/2,vmax=l_i+w_i/2)
    ax[1][1].imshow(ref_img[::-1,:,int(im_shape[2]/2)],cmap='gray',aspect=asp_sag,vmin=l_r-w_r/2,vmax=l_r+w_r/2)
    ax[1][2].imshow(mask_img[::-1,:,int(im_shape[2]/2)],cmap='gray',aspect=asp_sag)
    ax[1][3].imshow(diff[::-1,:,int(im_shape[2]/2)],cmap='RdBu',aspect=asp_sag,vmin=-0.3,vmax=0.3)

    for i in range(4):
        ax[0][i].set_xticklabels([])
        ax[0][i].set_yticklabels([])
        ax[0][i].set_title(titles[i].split('.')[0])

    ax[2][0].imshow(input_img[::-1,int(im_shape[1]/2),::-1],cmap='gray',aspect=asp_cor,vmin=l_i-w_i/2,vmax=l_i+w_i/2)
    ax[2][1].imshow(ref_img[::-1,int(im_shape[1]/2),::-1],cmap='gray',aspect=asp_cor,vmin=l_r-w_r/2,vmax=l_r+w_r/2)
    ax[2][2].imshow(mask_img[::-1,int(im_shape[1]/2),::-1],cmap='gray',aspect=asp_cor)
    ax[2][3].imshow(diff[::-1,int(im_shape[1]/2),::-1],cmap='RdBu',aspect=asp_cor,vmin=-0.3,vmax=0.3)

    for j in range(4):
        ax[2][j].set_xticklabels([])
        ax[2][j].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(output_path,transparent=False,facecolor='white',bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define fixed, moving and output filenames')
    parser.add_argument('operation', help='select operation to perform (register, convert, segment, mask_mr, mask_ct,resample, correct,overview, clean)')
    parser.add_argument('--f', help='fixed file path')
    parser.add_argument('--m', help='moving file path')
    parser.add_argument('--i', help='input file path (folder containing dicom series) for registration or resampling')
    parser.add_argument('--ii', help='2nd input file path')
    parser.add_argument('--o', help='output file path')
    parser.add_argument('--p', help='parameter file path (if not specified generate default)')
    parser.add_argument('--s', help='spacing used for resampling (size of the image will be adjusted accordingly)',
                        nargs='+', type=float)
    parser.add_argument('--r', help='radius for closing operation during masking')
    parser.add_argument('--mask_in', help='input mask to mask CT, CBCT or MR image')
    #    parser.add_argument('--mask_value',help = 'intensity value used outside mask')
    parser.add_argument('--mask_crop', help='mask to calculate bounding box for crop')
    args = parser.parse_args()

    if args.operation == 'register':
        if args.p is not None:
            register(args.f, args.m, read_parameter_map(args.p), args.o)
        # do something
        else:
            # do something else
            print('Please load a valid elastix parameter file!')
            # register(args.f, args.m, create_parameter_map(), args.o)
    elif args.operation == 'convert':
        convert_dicom_to_nifti(args.i, args.o)
    elif args.operation == 'resample':
        print(tuple(args.s))
        resample(args.i, args.o, tuple(args.s))
    elif args.operation == 'segment':
        segment(args.i, args.o, args.r)
    elif args.operation == 'correct':
        correct_mask_mr(args.i, args.ii, args.f, args.mask_crop, args.o)
        # mr, ct, params, mask, output mask
    elif args.operation == 'mask_mr':
        #        print('arg mask_value= '+ args.mask_value)
        mask_mr(args.i, args.mask_in, args.o)
    elif args.operation == 'mask_ct':
        #        print('arg mask_value= '+ args.mask_value)
        mask_ct(args.i, args.mask_in, args.o)
    elif args.operation == 'overview':
        generate_overview(args.i, args.ii, args.mask_in, args.o)
    elif args.operation == 'crop':
        crop(args.i, args.mask_crop, args.o)
    elif args.operation == 'clean':
        clean_border(args.i, args.o)
    else:
        print('check help for usage instructions')
              
        
import SimpleITK as sitk

def resize_image(input_image_path, output_image_path, pct_in, background_value, desired_size=(512, 512)):
    # Read the input image
    image = sitk.ReadImage(input_image_path)
    image_ar = sitk.GetArrayFromImage(image)
    
    pct = sitk.ReadImage(pct_in)
    
    # Get the size of the image
    original_size = image_ar.shape
    max_image_shape = desired_size[0]
    
    # Calculate the shift needed to keep the image at the center
    offsets = (original_size[0], int(np.floor(max_image_shape-original_size[1])/2.0), int(np.floor(max_image_shape-original_size[2])/2.0))
    # print(offsets)
    reshaped_img = np.ones((image_ar.shape[0], desired_size[0], desired_size[1]), dtype=np.float32) * float(background_value)
    reshaped_img[:,offsets[1]:offsets[1]+original_size[1],offsets[2]:offsets[2]+original_size[2]]=image_ar[:,:,:]
    
    plt.figure()
    plt.imshow(reshaped_img[90])
    plt.savefig('save_1.jpg')
    
    # Resize the image while keeping it at the center
    resampled_image = sitk.GetImageFromArray(reshaped_img)
    origin = pct.GetOrigin()
    spacing = pct.GetSpacing()
    print(origin,spacing)
    resampled_image.SetOrigin((origin[0]-(spacing[0]*offsets[1]),origin[1]-(spacing[1]*offsets[2]),origin[2]))
    # resampled_image.SetOrigin(pct.GetOrigin())
    # print(resampled_image.GetSize())
    # resampled_image.SetSpacing(pct.GetSpacing())
    # resampled_image.SetDirection(pct.GetDirection())
    
    # Write the resized image
    sitk.WriteImage(resampled_image, output_image_path,True)
    

def crop_z(input_image, mask_for_crop, output_image):
    image = sitk.ReadImage(input_image)
    mask = sitk.ReadImage(mask_for_crop)
    mask_np = sitk.GetArrayFromImage(mask)
    idx_nz = np.nonzero(mask_np)
    IP = [np.min(idx_nz[0]) , np.max(idx_nz[0]) ]
    cropped_image = image[:, :, IP[0]:IP[1]]
    sitk.WriteImage(cropped_image, output_image,True)
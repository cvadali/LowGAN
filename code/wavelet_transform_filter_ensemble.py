import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import pywt
import os
import argparse
import multiprocessing

"""
Code to remove stripes from the images using wavelet transform
Adapted from:
https://www.researchgate.net/publication/24419658_Stripe_and_ring_artifact_removal_with_combined_wavelet--Fourier_filtering

"""

def get_subject_list(subs_file):
    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)

    return full_subject_list


def xRemoveStripesVertical_axial(ima, decNum, wname, sigma):
    if ima.ndim == 2:
        return process_single_slice(ima, decNum, wname, sigma)
    else:
        # Process each slice/channel individually if ima has more than 2 dimensions
        result = np.empty_like(ima)
        for idx in range(ima.shape[-1]):
            result[:,:,idx] = process_single_slice(ima[:,:,idx], decNum, wname, sigma)
        return result

def xRemoveStripesVertical_coronal(ima, decNum, wname, sigma):
    if ima.ndim == 2:
        return process_single_slice(ima, decNum, wname, sigma)
    else:
        # Process each slice/channel individually if ima has more than 2 dimensions
        result = np.empty_like(ima)
        for idx in range(ima.shape[-2]):
            result[:,idx,:] = process_single_slice(ima[:,idx,:], decNum, wname, sigma)
        return result

def xRemoveStripesVertical_sagittal(ima, decNum, wname, sigma):
    if ima.ndim == 2:
        return process_single_slice(ima, decNum, wname, sigma)
    else:
        # Process each slice/channel individually if ima has more than 2 dimensions
        result = np.empty_like(ima)
        for idx in range(ima.shape[-3]):
            result[idx,:,:] = process_single_slice(ima[idx,:,:], decNum, wname, sigma)
        return result

def process_single_slice(slice_img, decNum, wname, sigma):
    # Save original shape
    original_shape = slice_img.shape

    # Pad to nearest power of two
    padded_shape = [2**np.ceil(np.log2(n)) for n in original_shape]
    padded_img = np.pad(slice_img, [(0, int(p - s)) for s, p in zip(original_shape, padded_shape)])

    # Wavelet decomposition
    Ch, Cv, Cd = [], [], []
    for _ in range(decNum):
        coeffs2 = pywt.dwt2(padded_img, wname)
        padded_img, (cH, cV, cD) = coeffs2
        Ch.append(cH)
        Cv.append(cV)
        Cd.append(cD)

    # FFT transform of horizontal frequency bands
    for ii in range(decNum):
        fCv = np.fft.fftshift(np.fft.fft2(Cv[ii]))
        my, mx = fCv.shape
        damp = 1 - np.exp(-np.arange(-my//2, my//2)**2 / (2 * sigma**2))
        fCv = fCv * damp[:, np.newaxis]
        Cv[ii] = np.fft.ifft2(np.fft.ifftshift(fCv)).real

    # Wavelet reconstruction
    nima = padded_img
    for ii in range(decNum - 1, -1, -1):
        nima = nima[:Ch[ii].shape[0], :Ch[ii].shape[1]]
        nima = pywt.idwt2((nima, (Ch[ii], Cv[ii], Cd[ii])), wname)

    # Crop to original size
    nima = nima[:original_shape[0], :original_shape[1]]

    return nima


def remove_stripes_axial(path_list, decNum, wname, damp_sigma):
    images = []
    for filename in [path_list]:
        img_loaded = nib.load(filename).get_fdata()
        img = xRemoveStripesVertical_coronal(img_loaded, decNum, wname, damp_sigma) # apply filter and skullstrip the filtered image
        img[img_loaded<=0] = 0
        img[img<=0] = 0
        images.append(img)
    
    # Stack images along a new dimension
    image_stack = np.stack(images, axis=-1)
    
    mean_image = np.mean(image_stack, axis=-1)

    return mean_image

def remove_stripes_coronal(path_list, decNum, wname, damp_sigma):
    images = []
    for filename in [path_list]:
        img_loaded = nib.load(filename).get_fdata()
        img = xRemoveStripesVertical_axial(img_loaded, decNum, wname, damp_sigma) # apply filter and skullstrip the filtered image
        img[img_loaded<=0] = 0
        img[img<=0] = 0
        images.append(img)
    
    # Stack images along a new dimension
    image_stack = np.stack(images, axis=-1)
    
    mean_image = np.mean(image_stack, axis=-1)

    return mean_image

def remove_stripes_sagittal(path_list, decNum, wname, damp_sigma):
    images = []
    for filename in [path_list]:
        img_loaded = nib.load(filename).get_fdata()
        img = xRemoveStripesVertical_coronal(img_loaded, decNum, wname, damp_sigma) # apply filter and skullstrip the filtered image
        img[img_loaded<=0] = 0
        img[img<=0] = 0
        images.append(img)
    
    # Stack images along a new dimension
    image_stack = np.stack(images, axis=-1)
    
    mean_image = np.mean(image_stack, axis=-1)

    return mean_image

# Save images
def save_image(data, reference_image, filename):
    new_image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    nib.save(new_image, filename)

# filter single volume of a single modality
def filter_single_volume(img_dir, sub, modality, output_dir, decNum, wname, damp_sigma):
    print(f'Processing {sub} {modality}')

    subject_path_axial = os.path.join(img_dir, f'{sub}_recon_{modality}_axial.nii.gz')
    subject_path_coronal = os.path.join(img_dir, f'{sub}_recon_{modality}_coronal.nii.gz')
    subject_path_sagittal = os.path.join(img_dir, f'{sub}_recon_{modality}_sagittal.nii.gz')

    try:
        img_axial = remove_stripes_axial(subject_path_axial, decNum, wname, damp_sigma)
        img_coronal = remove_stripes_coronal(subject_path_coronal, decNum, wname, damp_sigma)
        img_sagittal = remove_stripes_sagittal(subject_path_sagittal, decNum, wname, damp_sigma)

        img_final = (img_axial+img_coronal+img_sagittal)/3

        reference_image = nib.load(subject_path_axial)

        save_image(img_final,reference_image,os.path.join(output_dir,f'{sub}_recon_{modality}.nii.gz'))
    except Exception as e:
        print(e)

# filter in series
def filter_in_series(subs_file, path, output_dir, n_splits=12, decNum=8, wname='db25', damp_sigma=8):
    subjects = get_subject_list(subs_file)

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    for fold in folds:
        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_reshaped_coregistered_filtered')

        # create output_dir_fold if it doesn't exist
        if not os.path.exists(output_dir_fold):
            os.makedirs(output_dir_fold)

        img_dir_fold = os.path.join(path, fold, 'recon_niftis_reshaped_coregistered')

        for modality in ['flair','t1','t2']:
            for sub in subjects:
                filter_single_volume(img_dir_fold, sub, modality, output_dir_fold, decNum, wname, damp_sigma)

# filter in parallel
def filter_in_parallel(subs_file, path, output_dir, n_splits=12, decNum=8, wname='db25', damp_sigma=8):
    subjects = get_subject_list(subs_file)

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    max_processes = 10
    list_of_arguments = []

    for fold in folds:

        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_reshaped_coregistered_filtered')

        # create output_dir_fold if it doesn't exist
        if not os.path.exists(output_dir_fold):
            os.makedirs(output_dir_fold)

        img_dir_fold = os.path.join(path, fold, 'recon_niftis_reshaped_coregistered')

        for modality in ['flair','t1','t2']:
            for sub in subjects:
                list_of_arguments.append((img_dir_fold, sub, modality, output_dir_fold, decNum, wname, damp_sigma))

    with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(filter_single_volume, list_of_arguments)


# if script is run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove stripes from images using wavelet transform')

    parser.add_argument('-decNum', '--decNum', type=int, default=8, help='Decomposition level', required=False)
    parser.add_argument('-wname', '--wname', type=str, default='db25', help='Wavelet name', required=False)
    parser.add_argument('-damp_sigma', '--damp_sigma', type=int, default=8, help='Damping sigma for the stripe removal', required=False)
    parser.add_argument('-subs_file','--subs_file', type=str, help='File containing list of subjects', required=True)
    parser.add_argument('-path', '--path', type=str, help='Path to the directory containing the outputs', required=True)
    parser.add_argument('-output_dir', '--output_dir', type=str, help='Output directory', required=True)
    parser.add_argument('-n_splits', '--n_splits', type=int, default=12, help='Number of folds', required=False)
    parser.add_argument('-parallel', '--parallel', type=bool, default=False, help='Run in parallel', required=False)

    args = parser.parse_args()

    decNum = args.decNum
    wname = args.wname
    damp_sigma = args.damp_sigma
    subs_file = os.path.abspath(args.subs_file)
    path = os.path.abspath(args.path)
    output_dir = os.path.abspath(args.output_dir)
    n_splits = args.n_splits
    parallel = args.parallel

    if parallel:
        filter_in_parallel(subs_file, path, output_dir, n_splits, decNum, wname, damp_sigma)

    else:
        filter_in_series(subs_file, path, output_dir, n_splits, decNum, wname, damp_sigma)

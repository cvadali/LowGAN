import os
import ants
import nibabel as nib
import numpy as np
import argparse

# get list of subjects from subjects file
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

# reorient images to LPI
def reorient_images(path_source):
    # Load the image data
    image = ants.image_read(path_source)

    # Reorient the image to LPI
    reoriented_image = image.reorient_image2('LPI')

    return reoriented_image

# resample image to 1mm isotropic voxels
def resample_image(img):
    # use 1mm isotropic resampled image
    image = img.copy()

    # Calculate the new voxel spacing.
    new_spacing = [1, 1, 1]
  
    # Resample the image to 1mm isotropic.
    resampled_image = ants.resample_image(image, new_spacing, interp_type=0)

    return resampled_image

# rigid register images
def rigid_register_images(source_image, target_image, path_out):

    # Perform image registration
    registration_result = ants.registration(
        fixed=target_image,
        moving=source_image,
        type_of_transform="DenseRigid",
        verbose=False
    )

    # Get the registered image
    registered_image = registration_result['warpedmovout']

    # Save the registered image
    ants.image_write(registered_image, path_out)
    
    xfm = registration_result['fwdtransforms']
    
    return registered_image, xfm

# register images
def register_images(source_image, target_image):

    # Perform image registration
    registration_result = ants.registration(
        fixed=target_image,
        moving=source_image,
        type_of_transform="SyNRA",
        verbose=False
    )

    # Get the registered image
    registered_image = registration_result['warpedmovout']
    
    xfm = registration_result['fwdtransforms']
    
    return registered_image, xfm

# Apply registration matrix to an image
def apply_registration_matrix(source_image, target_image, registration_matrix):
    # Apply the registration matrix to the image
    transformed_image = ants.apply_transforms(fixed=target_image, moving=source_image, transformlist=registration_matrix)

    return transformed_image

# combine T1 and T2 for a single subject
def combine_T1_T2_single_sub(sub, source_data_dir, mni_template_path, gm_mask_path):
    print(f'Processing subject {sub}')

    anat_dir = os.path.join(source_data_dir, sub, 'session1_64mT', 'anat')

    derivatives_dir = os.path.join(source_data_dir, sub, 'derivatives')

    if not os.path.exists(derivatives_dir):
        os.makedirs(derivatives_dir)

    registered_images_dir = os.path.join(derivatives_dir, 'registered_images')

    if not os.path.exists(registered_images_dir):
        os.makedirs(registered_images_dir)

    sub_dir = os.path.join(source_data_dir, sub, 'derivatives', 'registered_images')

    # copy images in LPI to registered_images directory
    original_T1 = os.path.join(anat_dir, f'{sub}_T1.nii.gz')
    original_T2 = os.path.join(anat_dir, f'{sub}_T2.nii.gz')
    output_T1 = os.path.join(sub_dir, f'{sub}_lofi_T1_in_lofi_T1.nii.gz')
    output_T2 = os.path.join(sub_dir, f'{sub}_lofi_T2_in_lofi_T1.nii.gz')

    ants.image_write(resample_image(reorient_images(original_T1)), output_T1)
    ants.image_write(resample_image(reorient_images(original_T2)), output_T2)

    # register T2 to T1
    rigid_register_images(ants.image_read(output_T2), ants.image_read(output_T1), os.path.join(sub_dir, f'{sub}_lofi_T2_in_lofi_T1.nii.gz'))

    # Load the images
    T1_64mT = ants.image_read(os.path.join(sub_dir, f'{sub}_lofi_T1_in_lofi_T1.nii.gz'))
    T2_64mT = ants.image_read(os.path.join(sub_dir, f'{sub}_lofi_T2_in_lofi_T1.nii.gz'))
    mni_template = ants.image_read(mni_template_path)
    gm_mask = ants.image_read(gm_mask_path)

    print('Registering images')

    # Register the MNI template to the 64mT T1 image
    registered_mni_template_64mT, xfm_mni_64mT = register_images(mni_template, T1_64mT)

    # Apply the registration matrix to the GM mask
    registered_gm_mask_64mT = apply_registration_matrix(gm_mask, T1_64mT, xfm_mni_64mT)

    print('Computing scaling factors')

    # Extract the GM from the 64mT images
    gm_64mT_T1 = (registered_gm_mask_64mT * T1_64mT).numpy()
    gm_64mT_T2 = (registered_gm_mask_64mT * T2_64mT).numpy()

    # Compute median GM values
    M_G_T1_64mT = np.median(gm_64mT_T1[gm_64mT_T1 > 0])
    M_G_T2_64mT = np.median(gm_64mT_T2[gm_64mT_T2 > 0])

    # Compute the scaling factor
    s_64mT = M_G_T1_64mT / M_G_T2_64mT

    print('Combining images')

    T1_64mT = T1_64mT.numpy()
    T2_64mT = T2_64mT.numpy()

    # Combine the images
    combined_image_64mT = np.where((T1_64mT + (s_64mT * T2_64mT)) == 0, 0, (T1_64mT - (s_64mT * T2_64mT)) / (T1_64mT + (s_64mT * T2_64mT)))

    print('Saving images')

    # normalize to 0 to 1 range
    combined_image_64mT = combined_image_64mT - np.min(combined_image_64mT)
    combined_image_64mT = combined_image_64mT / ( np.max(combined_image_64mT) - np.min(combined_image_64mT) ) # set upper bound

    # Save the combined images
    T1_64mT_image = nib.load(os.path.join(sub_dir, f'{sub}_lofi_T1_in_lofi_T1.nii.gz'))

    combined_image_64mT_image = nib.Nifti1Image(combined_image_64mT, T1_64mT_image.affine, T1_64mT_image.header)

    nib.save(combined_image_64mT_image, os.path.join(sub_dir, f'{sub}_combined_lofi.nii.gz'))

    print(f'Finished processing subject {sub}')

# combine T1 and T2 for all subjects
def combine_T1_T2_all_subs(full_subject_list, source_data_dir, mni_template_path, gm_mask_path):
    for sub in full_subject_list:
        combine_T1_T2_single_sub(sub, source_data_dir, mni_template_path, gm_mask_path)

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Generate synthetic contrast-enhanced T1 image from T1 and T2 images')
    
    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with LowGAN subjects (each with T1 and T2 images)',
                        required=True,
                        )

    # MNI template path
    parser.add_argument('-mni_template','--mni_template',
                        help='Path to the MNI template',
                        required=True,
                        )

    # GM mask path
    parser.add_argument('-gm_mask','--gm_mask',
                        help='Path to the GM mask',
                        required=True,
                        )
    
    args = parser.parse_args()

    print(args)

    print('Starting')

    combine_T1_T2_all_subs(get_subject_list(os.path.abspath(args.subs_file)), os.path.abspath(args.data), os.path.abspath(args.mni_template), os.path.abspath(args.gm_mask))

    print('Finished')

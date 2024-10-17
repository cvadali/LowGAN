import os
import math
import numpy as np
import ants
import nibabel as nib
import argparse
import multiprocessing

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

# unpad from cube
def unpad_cube(padded_img, orig_img):
    # this function un-pads the image to its original size
    x, y, z = orig_img.shape
    x_p, y_p, z_p = padded_img.shape
    
    to_remove_x = (x_p - x) // 2
    to_remove_y = (y_p - y) // 2
    to_remove_z = (z_p - z) // 2
    
    if x_p >= x:
        unpadded_x = padded_img[to_remove_x:x+to_remove_x]
    else:
        pad_width_x = ((x - x_p) // 2, (x - x_p) // 2 + (x - x_p) % 2)
        unpadded_x = np.pad(padded_img, pad_width=((pad_width_x[0], pad_width_x[1]), (0, 0), (0, 0)), mode='constant')
    
    if y_p >= y:
        unpadded_y = unpadded_x[:, to_remove_y:y+to_remove_y]
    else:
        pad_width_y = ((y - y_p) // 2, (y - y_p) // 2 + (y - y_p) % 2)
        unpadded_y = np.pad(unpadded_x, pad_width=((0, 0), (pad_width_y[0], pad_width_y[1]), (0, 0)), mode='constant')
    
    if z_p >= z:
        unpadded = unpadded_y[:, :, to_remove_z:z+to_remove_z]
    else:
        pad_width_z = ((z - z_p) // 2, (z - z_p) // 2 + (z - z_p) % 2)
        unpadded = np.pad(unpadded_y, pad_width=((0, 0), (0, 0), (pad_width_z[0], pad_width_z[1])), mode='constant')
    
    return unpadded

# unpad a single volume
def unpad_single_volume(padded_img_path, orig_img_path, output_path):
    # load the padded image
    padded_img = nib.load(padded_img_path)

    # load the original image
    orig_img = nib.load(orig_img_path)

    # unpad the image
    unpadded_img = unpad_cube(padded_img.get_fdata(), orig_img.get_fdata())

    # save the image
    nib.save(nib.Nifti1Image(unpadded_img, padded_img.affine), output_path)

    # reorient the image
    saved_image = ants.image_read(output_path)

    reoriented_image = saved_image.reorient_image2('LPI')

    ants.image_write(reoriented_image, output_path)

# unpad T1, T2, FLAIR for a single subject
def unpad_subject(subject, padded_dir, orig_dir, output_dir):
    print('Unpadding subject: ', subject)

    # unpad T1
    unpad_single_volume(os.path.join(padded_dir, subject + '_recon_t1.nii.gz'), 
                        os.path.join(orig_dir, subject, 'derivatives', 'registered_images', subject + '_lofi_T1_in_lofi_T1_skullstripped.nii.gz'), 
                        os.path.join(output_dir, subject + '_LowGAN_T1.nii.gz'))

    # unpad T2
    unpad_single_volume(os.path.join(padded_dir, subject + '_recon_t2.nii.gz'), 
                        os.path.join(orig_dir, subject, 'derivatives', 'registered_images', subject + '_lofi_T2_in_lofi_T1_skullstripped.nii.gz'), 
                        os.path.join(output_dir, subject + '_LowGAN_T2.nii.gz'))

    # unpad FLAIR
    unpad_single_volume(os.path.join(padded_dir, subject + '_recon_flair.nii.gz'), 
                        os.path.join(orig_dir, subject, 'derivatives', 'registered_images', subject + '_lofi_combined_in_lofi_T1_skullstripped.nii.gz'), 
                        os.path.join(output_dir, subject + '_LowGAN_FLAIR.nii.gz'))

    if os.path.exists(os.path.join(output_dir, subject + '_LowGAN_T1.nii.gz')) and \
       os.path.exists(os.path.join(output_dir, subject + '_LowGAN_T2.nii.gz')) and \
       os.path.exists(os.path.join(output_dir, subject + '_LowGAN_FLAIR.nii.gz')):
        print('Unpadding subject: ', subject, ' done!')
    
    else:
        print('Unpadding subject: ', subject, ' failed!')

# unpad in series
def unpad_series(subjects, padded_dir, orig_dir, output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for subject in subjects:
        unpad_subject(subject, padded_dir, orig_dir, output_dir)

# unpad in parallel
def unpad_parallel(subjects, padded_dir, orig_dir, output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    max_processes = 10
    list_of_arguments = []

    for subject in subjects:
        list_of_arguments.append((subject, padded_dir, orig_dir, output_dir))

    with multiprocessing.Pool(max_processes) as pool:
        pool.starmap(unpad_subject, list_of_arguments)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unpad the images from a cube to the original shape')
    parser.add_argument('-subs_file', '--subs_file', type=str, help='File containing list of subjects')
    parser.add_argument('-padded_dir', '--padded_dir', type=str, help='Directory containing padded images')
    parser.add_argument('-orig_dir', '--orig_dir', type=str, help='Directory containing original images')
    parser.add_argument('-output_dir', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-parallel', '--parallel', help='(OPTIONAL) Run in parallel', action='store_true')

    args = parser.parse_args()

    # get the list of subjects
    subjects = get_subject_list(os.path.abspath(args.subs_file))

    if args.parallel:
        unpad_parallel(subjects, os.path.abspath(args.padded_dir), os.path.abspath(args.orig_dir), os.path.abspath(args.output_dir))

    else:
        unpad_series(subjects, os.path.abspath(args.padded_dir), os.path.abspath(args.orig_dir), os.path.abspath(args.output_dir))



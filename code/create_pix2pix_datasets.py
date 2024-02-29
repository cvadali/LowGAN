import numpy as np
import os
import cv2
import sys
import ants
import argparse
import multiprocessing
import time

# compute centroid of mask
def calculate_mask_centroid(image):
    # Load the image data
    image_mask = (image>0)

    # Calculate the centroid coordinates
    indices = np.indices(image_mask.shape)
    centroid = np.mean(indices[:, image_mask.astype(bool)], axis=1)

    return centroid

# pad with zeros
def zero_pad(img, pad=50):
    x,y,z = img.shape
    img_padded = np.ones((x+pad,y+pad, z+pad))*img[0,0,0]
    img_padded[pad//2:pad//2 + x, pad//2:pad//2 + y, pad//2:pad//2 + z] = img
    return img_padded

# pad into a cube
def make_cube(img):
    # this function zero pads the image until it becomes a cube
    x,y,z = img.shape
    max_dim = np.max(img.shape)
    
    to_add_x = (max_dim-x)//2
    to_add_y = (max_dim-y)//2
    to_add_z = (max_dim-z)//2
    
    zero_padded = np.ones((x+(to_add_x*2), y+(to_add_y*2), z+(to_add_z*2)))*img[0,0,0]
    
    zero_padded[to_add_x:x+to_add_x, to_add_y:y+to_add_y, to_add_z:z+to_add_z] = img
    
    return zero_padded

# normalize and pad to a cube
def norm_zero_to_one(vol3D):
    '''Normalize a 3D volume to zero to one range

    Parameters:
      vol3D (3D numpy array): 3D image volume
    '''

    # normalize to 0 to 1 range
    vol3D = vol3D - np.min(vol3D) # set lower bound
    vol3D = vol3D / ( np.max(vol3D) - np.min(vol3D) ) # set upper bound

    return make_cube(vol3D)

# load subject images
def load_subject_images(data_source, subject, plane, hifi_or_lofi):
    # if 3T images
    if hifi_or_lofi == 'hifi':
        t1 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_hifi_T1_skullstripped.nii.gz'))
        t2 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_hifi_T2_skullstripped.nii.gz'))
        flair = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_hifi_FLAIR_skullstripped.nii.gz'))

    # if 64mT images
    elif hifi_or_lofi == 'lofi':
        t1 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_T1_in_hifi_T1_skullstripped.nii.gz'))
        t2 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_T2_in_hifi_T2_skullstripped.nii.gz'))
        flair = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_FLAIR_in_hifi_FLAIR_skullstripped.nii.gz'))
    
    else:
        print('Must input either hifi (for 3T images) or lofi (for 64mT images) for hifi_or_lofi')
        sys.exit()

    subject_dict = []

    if t1.shape[0]==np.min(t1.shape):
        flip=True
    else:
        flip=False

    # multiply by 255 for RGB conversion
    if flip==False:
        subject_dict.append(norm_zero_to_one((t1).numpy().swapaxes(0,1)) * 255)
        subject_dict.append(norm_zero_to_one((t2).numpy().swapaxes(0,1)) * 255)
        subject_dict.append(norm_zero_to_one((flair).numpy().swapaxes(0,1)) * 255)
    else:
        print('flip')
        subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((t1).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)
        subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((t2).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)
        subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((flair).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)


    subject_dict = np.array(subject_dict)

    if plane == 'axial':
        subject_dict.swapaxes(1,2)
    
    elif plane == 'coronal':
        subject_dict = subject_dict
    
    elif plane == 'sagittal':
        subject_dict = np.flip(np.rot90(subject_dict, k=3, axes=(1,2)), axis=2)
    
    else:
        print('Must enter valid plane')
        sys.exit()

    return subject_dict


# Function for concatenating the T1, T2 and FLAIR
def concat_t1_t2_flair(t1,t2,flair):
    return np.concatenate([t1[:,:,np.newaxis],t2[:,:,np.newaxis],flair[:,:,np.newaxis]],axis=2)

# concatenate hifi and lofi images next to each other
def concat_hifi_lofi(hifi_array, lofi_array):
    hifi_and_lofi_array = np.concatenate([hifi_array, lofi_array], 1)

    return hifi_and_lofi_array

# create dataset for a single sub for a given plane
def create_dataset_single_sub_single_plane(subID, data_source, outdir, plane):
    print(f'Processing: {subID} {plane}')

    # Check if the folder where the .png files will be output exists 
    # if not, create it
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    # preprocess and save out paired lofi/hifi datasets of PNG files
    # select appropriate volume
    array_hifi = load_subject_images(data_source, subID, plane, 'hifi')

    array_lofi = load_subject_images(data_source, subID, plane, 'lofi')

    output_path = os.path.join(outdir, f'dataset_{plane}_pix2pix')

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    z = array_hifi.shape[2] # process along the z-axis


    for i in range(0,z):
        filename = subID + '_' + str(i) + '.png'

        filepath = os.path.join(output_path, filename)

        # slice differently depending on plane
        if plane == 'axial':
            hifi_png_array = concat_t1_t2_flair(array_hifi[0,i,:,:], array_hifi[1,i,:,:], array_hifi[2,i,:,:])
            lofi_png_array = concat_t1_t2_flair(array_lofi[0,i,:,:], array_lofi[1,i,:,:], array_lofi[2,i,:,:])
        elif plane == 'coronal':
            hifi_png_array = concat_t1_t2_flair(array_hifi[0,:,i,:], array_hifi[1,:,i,:], array_hifi[2,:,i,:])
            lofi_png_array = concat_t1_t2_flair(array_lofi[0,:,i,:], array_lofi[1,:,i,:], array_lofi[2,:,i,:])
        elif plane == 'sagittal':
            hifi_png_array = concat_t1_t2_flair(array_hifi[0,:,:,i], array_hifi[1,:,:,i], array_hifi[2,:,:,i])
            lofi_png_array = concat_t1_t2_flair(array_lofi[0,:,:,i], array_lofi[1,:,:,i], array_lofi[2,:,:,i])
        else:
            print('Must specify a plane')
            sys.exit()
        
        # combine hifi and lofi images side by side
        # hifi on left, lofi on right
        combined_hifi_lofi_png_array = concat_hifi_lofi(hifi_png_array, lofi_png_array)

        # save combined hifi lofi image
        cv2.imwrite(filepath, combined_hifi_lofi_png_array)

# create dataset for a given plane (axial, coronal, or sagittal)
def create_dataset_single_plane(subs_file, data_source, outdir, plane):

    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)

    # process all subjects
    for subID in full_subject_list:
        create_dataset_single_sub_single_plane(subID, data_source, outdir, plane)

# iterate over each plane
def create_dataset_all_planes(subs_file, data_source, outdir):
    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:
        print(plane)
        create_dataset_single_plane(subs_file, data_source, outdir, plane)
    
    print('Finished')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Create pix2pix datasets for LowGAN network')
    
    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with LowGAN subjects (each with t1, t2, and flair images)',
                        required=True,
                        )
    
    # model name
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the images',
                        required=True,
                        )
    
    # (OPTIONAL) run in parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel',
                        required=False,
                        default=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    if bool(args.parallel) == True:
        print('Processing in parallel')
        
        max_processes = 20

        # file containing subjects
        subject_file = open(args.subs_file, 'r')

        # list of all subjects
        full_subject_list = []

        # convert to list
        for line in subject_file:
            line = line.split('\n')[0]
            full_subject_list.append(line)
        
        list_of_arguments = []

        planes = ['axial', 'coronal', 'sagittal']
        
        # iterate for each subject
        for sub in full_subject_list:
            for plane in planes:
                list_of_arguments.append((sub, args.data, args.output_dir, plane))
        
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(create_dataset_single_sub_single_plane, list_of_arguments)
        
        print('Finished parallel processing')

    else:
        create_dataset_all_planes(
            subs_file=args.subs_file,
            data_source=args.data,
            outdir=args.output_dir
        )

    print('Finished')

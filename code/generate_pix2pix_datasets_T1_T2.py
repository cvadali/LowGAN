import numpy as np
import os
import cv2
import sys
import math
import ants
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

# pad into a cube
def make_cube(img):
    # this function zero pads the image until it becomes a cube
    x,y,z = img.shape
    max_dim = np.max(img.shape)
    
    to_add_x = (max_dim - x) / 2
    to_add_y = (max_dim - y) / 2
    to_add_z = (max_dim -  z) / 2
    
    zero_padded = np.ones((int(x+(to_add_x*2)), int(y+(to_add_y*2)), int(z+(to_add_z*2))))*img[0,0,0]
    
    zero_padded[math.floor(to_add_x):x+math.floor(to_add_x), 
                math.floor(to_add_y):y+math.floor(to_add_y), 
                math.floor(to_add_z):z+math.floor(to_add_z)] = img
    
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
def load_subject_images(data_source, subject, plane):
    t1 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_T1_in_lofi_T1_skullstripped.nii.gz'))
    t2 = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_T2_in_lofi_T1_skullstripped.nii.gz'))
    t1ce = ants.image_read(os.path.join(data_source,subject, 'derivatives', 'registered_images', f'{subject}_lofi_combined_in_lofi_T1_skullstripped.nii.gz'))
    

    subject_dict = []

    subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((t1).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)
    subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((t2).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)
    subject_dict.append(norm_zero_to_one(np.flip(np.fliplr(np.rot90((t1ce).numpy().swapaxes(0,2))).swapaxes(0,1), axis=2)) * 255)


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


# Function for concatenating the T1, T2 and T1ce
def concat_t1_t2_t1ce(t1,t2,t1ce):
    return np.concatenate([t1[:,:,np.newaxis],t2[:,:,np.newaxis],t1ce[:,:,np.newaxis]],axis=2)

# concatenate the same lofi images next to each other
def concat_lofi_lofi(lofi_array):
    lofi_and_lofi_array = np.concatenate([lofi_array, lofi_array], 1)

    return lofi_and_lofi_array

# generate dataset for a single sub for a given plane
def generate_dataset_single_sub_single_plane(subID, data_source, outdir, plane):
    print(f'Processing: {subID} {plane}')

    # preprocess and save out paired lofi/lofi datasets of PNG files
    # select appropriate volume

    array_lofi = load_subject_images(data_source, subID, plane)

    output_path = os.path.join(outdir, f'dataset_{plane}_pix2pix', 'test')

    z = max(array_lofi.shape) # maximum dimension


    for i in range(0,z):
        filename = subID + '_' + str(i) + '.png'

        filepath = os.path.join(output_path, filename)

        # slice differently depending on plane
        if plane == 'axial':
            lofi_png_array = concat_t1_t2_t1ce(array_lofi[0,i,:,:], array_lofi[1,i,:,:], array_lofi[2,i,:,:])
        elif plane == 'coronal':
            lofi_png_array = concat_t1_t2_t1ce(array_lofi[0,:,i,:], array_lofi[1,:,i,:], array_lofi[2,:,i,:])
        elif plane == 'sagittal':
            lofi_png_array = concat_t1_t2_t1ce(array_lofi[0,:,:,i], array_lofi[1,:,:,i], array_lofi[2,:,:,i])
        else:
            print('Must specify a plane')
            sys.exit()
        
        # combine lofi and lofi images side by side
        # lofi on left, lofi on right
        # because pix2pix requires this
        combined_lofi_lofi_png_array = concat_lofi_lofi(lofi_png_array)

        # save combined hifi lofi image
        cv2.imwrite(filepath, combined_lofi_lofi_png_array)

# generate dataset for a given plane (axial, coronal, or sagittal)
def generate_dataset_single_plane(subs_file, data_source, outdir, plane):

    full_subject_list = get_subject_list(subs_file)

    # process all subjects
    for sub in full_subject_list:
        generate_dataset_single_sub_single_plane(sub, data_source, outdir, plane)

# iterate over each plane
def generate_dataset_all_planes(subs_file, data_source, outdir):
    # Check if the folder where the .png files will be output exists 
    # if not, create it
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    
    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:
        output_path = os.path.join(outdir, f'dataset_{plane}_pix2pix')

        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        
        print(plane)
        generate_dataset_single_plane(subs_file, data_source, outdir, plane)
    
    print('Finished')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Generate pix2pix datasets for LowGAN network')
    
    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with LowGAN subjects (each with t1, t2, and combined images)',
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
        
        max_processes = 10

        # file containing subjects
        full_subject_list = get_subject_list(os.path.abspath(args.subs_file))
        
        list_of_arguments = []

        planes = ['axial', 'coronal', 'sagittal']

        if os.path.exists(os.path.abspath(args.output_dir)) == False:
            os.makedirs(os.path.abspath(args.output_dir))
        
        # iterate for each subject
        for sub in full_subject_list:
            for plane in planes:
                output_path = os.path.join(os.path.abspath(args.output_dir), f'dataset_{plane}_pix2pix')

                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)

                output_path_test = os.path.join(output_path, 'test')

                if os.path.exists(output_path_test) == False:
                    os.makedirs(output_path_test)
                    
                list_of_arguments.append((sub, os.path.abspath(args.data), os.path.abspath(args.output_dir), plane))
        
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(generate_dataset_single_sub_single_plane, list_of_arguments)
        
        print('Finished parallel processing')

    else:
        generate_dataset_all_planes(
            subs_file=os.path.abspath(args.subs_file),
            data_source=os.path.abspath(args.data),
            outdir=os.path.abspath(args.output_dir)
        )

    print('Finished')

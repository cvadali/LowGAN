import os
import antspynet as apn
import ants
import glob
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

# register images
def register_images(source_image, target_image, path_out):

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

# resample image to 1mm isotropic voxels
def resample_image(img):
    # use 1mm isotropic resampled image
    image = img.copy()

    # Calculate the new voxel spacing.
    new_spacing = [1, 1, 1]
  
    # Resample the image to 1mm isotropic.
    resampled_image = ants.resample_image(image, new_spacing, interp_type=0)

    return resampled_image

def register_and_skullstrip_each_subject(full_subject_list, data_dir):
    
    modalities = ['T1', 'T2', 'combined']

    for sub in full_subject_list:
        try:

            print(f'Processing: {sub}')

            # directory for subject
            sub_dir = os.path.join(data_dir, sub)

            # derivatives for subject
            derivatives_dir = os.path.join(sub_dir, 'derivatives')

            if os.path.exists(derivatives_dir) == False:
                os.makedirs(derivatives_dir)

            # directory within derivatives containing registered images
            registered_images_dir = os.path.join(derivatives_dir, 'registered_images')

            if os.path.exists(registered_images_dir) == False:
                os.makedirs(registered_images_dir)
            
            for modality in modalities:
                print(f'Processing: {sub} {modality}')

                print('registering to LPI')

                if modality == 'T1':
                    reoriented_image_lofi = ants.image_read(os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_lofi_T1.nii.gz'))

                elif modality == 'T2':
                    reoriented_image_lofi = ants.image_read(os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_lofi_T1.nii.gz'))

                elif modality == 'combined':
                    reoriented_image_lofi = ants.image_read(os.path.join(registered_images_dir, f'{sub}_combined_lofi.nii.gz'))

                # register lofi to lofi T1
                reg_image_lofi, xfm_lofi = register_images(reoriented_image_lofi,
                                                           ants.image_read(os.path.join(registered_images_dir, f'{sub}_lofi_T1_in_lofi_T1.nii.gz')),
                            os.path.join(registered_images_dir,f'{sub}_lofi_{modality}_in_lofi_T1.nii.gz'))
                
                print('skullstripping')

                # skullstrip the lofi image
                if modality == 'T1':
                    brain_extracted_T1_lofi = apn.brain_extraction(ants.image_read(os.path.join(registered_images_dir, 
                                                                        f'{sub}_lofi_{modality}_in_lofi_T1.nii.gz')),
                                                                        modality=modality.lower())

                elif modality == 'T2':
                    brain_extracted_T1_lofi = apn.brain_extraction(ants.image_read(os.path.join(registered_images_dir, 
                                                                        f'{sub}_lofi_{modality}_in_lofi_T1.nii.gz')),
                                                                        modality=modality.lower())

                elif modality == 'combined':
                    brain_extracted_T1_lofi = apn.brain_extraction(ants.image_read(os.path.join(registered_images_dir, 
                                                                        f'{sub}_lofi_T1_in_lofi_T1.nii.gz')),
                                                                        modality='T1'.lower())
                
                print('save skullstripping')

                ants.image_write(resample_image(reg_image_lofi)*brain_extracted_T1_lofi,os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_lofi_T1_skullstripped.nii.gz'))

                print('coregister skullstripped image')

                # coregister lofi brains to lofi T1
                register_images(ants.image_read(os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_lofi_T1_skullstripped.nii.gz')), 
                                ants.image_read(os.path.join(registered_images_dir, f'{sub}_lofi_T1_in_lofi_T1_skullstripped.nii.gz')), 
                                os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_lofi_T1_skullstripped.nii.gz'))

                # save the transform from lofi to lofi t1
                red_xfm_lofi = ants.read_transform(xfm_lofi[0])
                ants.write_transform(red_xfm_lofi, os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_to_lofi_t1_xfm.mat'))
                
                print('save brain mask')
                # save the brain mask for the subjects
                if modality == 'T1':
                    ants.image_write(brain_extracted_T1_lofi, os.path.join(registered_images_dir, f'{sub}_lofi_brainmask.nii.gz'))
            
        except Exception as e:
            print(e)

# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reorient, register, and skullstrip 64mT images in preparation for running through LowGAN')

     # file containing name of subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with source data',
                        required=True,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    register_and_skullstrip_each_subject(
        full_subject_list=full_subject_list,
        data_dir=args.data
    )

    print('Finished')

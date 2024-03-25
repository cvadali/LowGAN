import os
import antspynet as apn
import ants
import glob
import argparse

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

def get_hifi_path(subject, directory, modality):
    path = os.path.join(directory,subject,'session1_3T','anat',subject+'_'+modality+'.ni*')
    return glob.glob(path)[0]

def get_lofi_path(subject, directory, modality):
    path = os.path.join(directory,subject,'session1_64mT','anat',subject+'_'+modality+'.ni*')
    return glob.glob(path)[0]

def reorient_images(path_source):
    # Load the image data
    image = ants.image_read(path_source)

    # Reorient the image to LPI
    reoriented_image = image.reorient_image2('LPI')

    return reoriented_image


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

def resample_image(img):
    # use 1mm isotropic resampled image
    image = img.copy()

    # Get the original voxel spacing.
    original_spacing = image.spacing

    # Calculate the new voxel spacing.
    new_spacing = [1, 1, 1]
  
    # Resample the image to 1mm isotropic.
    resampled_image = ants.resample_image(image, new_spacing, interp_type=0)

    return resampled_image


def register_and_skullstrip_each_subject(subs_file, data_dir):
    full_subject_list = get_subject_list(subs_file)
    
    modalities = ['T1', 'T2', 'FLAIR']

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

                # reorient hifi to LPI
                reoriented_image_hifi = reorient_images(get_hifi_path(sub, data_dir, modality))

                # register hifi to hifi
                reg_image_hifi, xfm_hifi = register_images(reoriented_image_hifi,ants.read(get_hifi_path(sub, data_dir, 'T1')),
                            os.path.join(registered_images_dir,f'{sub}_hifi_{modality}_in_hifi_T1.nii.gz'))
                
                # reorient lofi to LPI
                reoriented_image_lofi = reorient_images(get_lofi_path(sub, data_dir, modality))

                # register lofi to hifi
                reg_image_lofi, xfm_lofi = register_images(reoriented_image_lofi,ants.read(get_hifi_path(sub, data_dir, 'T1')),
                            os.path.join(registered_images_dir,f'{sub}_lofi_{modality}_in_hifi_T1.nii.gz'))

                # skullstrip the lofi image
                brain_extracted_T1 = apn.brain_extraction(ants.image_read(get_hifi_path(sub, data_dir, 'T1')),modality='T1'.lower())
                ants.image_write(ants.resample_image_to_target(reg_image_lofi,brain_extracted_T1)*brain_extracted_T1,os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_hifi_{modality}_skullstripped.nii.gz'))
                
                # skullstrip the hifi image
                ants.image_write(ants.resample_image_to_target(reg_image_hifi,brain_extracted_T1)*brain_extracted_T1,os.path.join(registered_images_dir, f'{sub}_hifi_{modality}_skullstripped.nii.gz'))

                # coregister lofi brains to hifi brains
                register_images(ants.read(os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_hifi_{modality}_skullstripped.nii.gz')), 
                                ants.read(os.path.join(registered_images_dir, f'{sub}_hifi_T1_skullstripped.nii.gz')), 
                                os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_in_hifi_{modality}_skullstripped.nii.gz'))
                
                # coregister hifi brains to hifi brains
                register_images(ants.read(os.path.join(registered_images_dir, f'{sub}_hifi_{modality}_skullstripped.nii.gz')), 
                                ants.read(os.path.join(registered_images_dir, f'{sub}_hifi_T1_skullstripped.nii.gz')), 
                                os.path.join(registered_images_dir, f'{sub}_hifi_{modality}_skullstripped.nii.gz'))
                
                # save the non-skullstripped hifi image in the same space as the hifi t1
                ants.image_write(ants.resample_image_to_target(reg_image_hifi,brain_extracted_T1),os.path.join(registered_images_dir, f'{sub}_hifi_{modality}_not_skullstripped.nii.gz'))

                # save the transform from hifi to hifi t1
                
                red_xfm_hifi = ants.read_transform(xfm_hifi[0])
                ants.write_transform(red_xfm_hifi, os.path.join(registered_images_dir, f'{sub}_hifi_{modality}_to_hifi_t1_xfm.mat'))

                # save the transform from lofi to hifi t1
                
                red_xfm_lofi = ants.read_transform(xfm_lofi[0])
                ants.write_transform(red_xfm_lofi, os.path.join(registered_images_dir, f'{sub}_lofi_{modality}_to_hifi_t1_xfm.mat'))
                
                # save the brain mask for the subjects
                ants.image_write(brain_extracted_T1, os.path.join(registered_images_dir, f'{sub}_brainmask.nii.gz'))
            
        except Exception as e:
            print(e)


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Register and skullstrip 3T and 64mT images')

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

    register_and_skullstrip_each_subject(
        subs_file=args.subs_file,
        data_dir=args.data
    )

    print('Finished')



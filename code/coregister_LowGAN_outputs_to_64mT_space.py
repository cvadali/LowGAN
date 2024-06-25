import os
import ants
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

# register a single image to 64mT T1
def register_to_64mT_T1(subject, modality, lofi_dir, recon_dir, out_dir):
    print(f'Processing: {subject} {modality}')

    # get lofi path
    lofi_path = os.path.join(lofi_dir, subject, 'derivatives', 'registered_images', f'{subject}_lofi_T1_in_lofi_T1_skullstripped.nii.gz')
    recon_path = os.path.join(recon_dir, f'{subject}_recon_{modality}.nii.gz')

    # load images
    lofi = ants.image_read(lofi_path)
    recon = ants.image_read(recon_path)

    # register images
    out_path = os.path.join(out_dir, f'{subject}_recon_{modality}.nii.gz')

    registered_image, xfm = register_images(recon, lofi, out_path)

    if os.path.exists(out_path):
        print(f'Saved: {out_path}')
    
    else:
        print(f'Failed to save: {out_path}')
        sys.exit()

# register all modalities for all subjects
def register_all_modalities(subjects, lofi_dir, recon_dir, out_dir):
    modalities = ['t1', 't2', 'flair']

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    for subject in subjects:
        for modality in modalities:
            register_to_64mT_T1(subject, modality, lofi_dir, recon_dir, out_dir)

# if script is run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register LowGAN outputs to 64mT space')
    parser.add_argument('-subs_file', '--subs_file', help='File containing list of subjects', required=True)
    parser.add_argument('-lofi_dir', '--lofi_dir', help='Directory containing 64mT images', required=True)
    parser.add_argument('-recon_dir', '--recon_dir', help='Directory containing LowGAN outputs', required=True)
    parser.add_argument('-output_dir', '--output_dir', help='Output directory for registered images', required=True)

    args = parser.parse_args()

    print('Starting')

    # get list of subjects
    subjects = get_subject_list(os.path.abspath(args.subs_file))

    # register all modalities for all subjects
    register_all_modalities(subjects, os.path.abspath(args.lofi_dir), os.path.abspath(args.recon_dir), os.path.abspath(args.output_dir))

    print('Finished')


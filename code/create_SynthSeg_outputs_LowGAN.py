import os
import argparse
import multiprocessing

# generate SynthSeg segmentation of a single LowGAN T1 image
def create_SynthSeg_segmentation_single_image(subject, SynthSeg_dir, image_path, output_dir):
    print(f'Processing: {subject}')

    # path to script
    script = os.path.join(SynthSeg_dir, 'scripts', 'commands', 'SynthSeg_predict.py')

    # path to output segmentation file
    output_path = os.path.join(output_dir, f'{subject}_LowGAN_T1_SynthSeg.nii.gz')

    # run the command
    os.system(f'python {script} --i {image_path} --o {output_path}')

    success = os.path.exists(output_path)

    return success

# iterate for every subject in list
def iterate_for_all_subs(SynthSeg_dir, images_dir, output_dir, n_splits=12):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    for fold in folds:
        # all stage 2 outputs
        all_images = os.listdir(os.path.join(images_dir, fold, 'recon_niftis_stage2'))

        full_subject_list = []

        for image in all_images:
            subject = image.split('_')[0]
            full_subject_list.append(subject)
        
        # remove duplicates
        full_subject_list = list(set(full_subject_list))
    
        # iterate for each subject
        for sub in full_subject_list:
            # path to T1 image
            image_path = os.path.join(images_dir, fold, 'recon_niftis_stage2', f'{sub}_recon_t1.nii.gz')

            success = create_SynthSeg_segmentation_single_image(sub, SynthSeg_dir, image_path, output_dir)

            if success == True:
                print(f'{sub} successful')
            
            else:
                print(f'{sub} failed')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Create SynthSeg segmentations of LowGAN T1 images')
    
    # SynthSeg directory
    parser.add_argument('-synthseg_dir','--synthseg_dir',
                        help='Directory with SynthSeg installation',
                        required=True,
                        )

    # Images directory
    parser.add_argument('-images_dir','--images_dir',
                        help='Directory with LowGAN image outputs',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the SynthSeg segmentations',
                        required=True,
                        )
    
    # (OPTIONAL) Number of folds
    parser.add_argument('-n_splits', '--n_splits',
                        help='Number of folds',
                        default=12,
                        required=False,
                        )
    
    # (OPTIONAL) process in parallel
    parser.add_argument('-parallel', '--parallel',
                        help='Run reconstruction in parallel',
                        default=False,
                        required=False,
                        )
    
    args = parser.parse_args()

    print(args)

    print('Starting')

    # process in parallel, 5 at a time
    if bool(args.parallel) == True:
        max_processes = 5

        list_of_arguments = []

        # get list of folds
        folds = [f'fold_{fold}' for fold in range(0, int(args.n_splits))]

        for fold in folds:
            # all stage 2 outputs
            all_images = os.listdir(os.path.join(os.path.abspath(args.images_dir), fold, 'recon_niftis_stage2'))

            full_subject_list = []

            for image in all_images:
                subject = image.split('_')[0]
                full_subject_list.append(subject)
            
            # remove duplicates
            full_subject_list = list(set(full_subject_list))

            # iterate for each subject
            for sub in full_subject_list:
                # path to T1 image
                image_path = os.path.join(os.path.abspath(args.images_dir), fold, 'recon_niftis_stage2', f'{sub}_recon_t1.nii.gz')

                # create list of argument tuples
                list_of_arguments.append((sub, os.path.abspath(args.synthseg_dir), image_path, os.path.abspath(args.output_dir)))

        if os.path.exists(os.path.abspath(args.output_dir)) == False:
            os.makedirs(os.path.abspath(args.output_dir))
        
        # spawn up 5 processes at once
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(create_SynthSeg_segmentation_single_image, list_of_arguments)
        
        print('Finished parallel SynthSeg segmentation')
    
    # run in series
    else:
        iterate_for_all_subs(
            SynthSeg_dir=os.path.abspath(args.synthseg_dir),
            images_dir=os.path.abspath(args.images_dir),
            output_dir=os.path.abspath(args.output_dir),
            n_splits=int(args.n_splits)
        )
    
    print('Finished')


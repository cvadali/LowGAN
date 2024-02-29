import os
import argparse
import multiprocessing

# generate SynthSeg segmentation of a single 3T T1 image
def create_SynthSeg_segmentation_single_image(subject, SynthSeg_dir, image_path, output_dir):
    print(f'Processing: {subject}')

    # path to script
    script = os.path.join(SynthSeg_dir, 'scripts', 'commands', 'SynthSeg_predict.py')

    # path to output segmentation file
    output_path = os.path.join(output_dir, f'{subject}_3T_T1_SynthSeg.nii.gz')

    # run the command
    os.system(f'python {script} --i {image_path} --o {output_path}')

    success = os.path.exists(output_path)

    return success

# iterate for every subject in list
def iterate_for_all_subs(subs_file, SynthSeg_dir, images_dir, output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)
    
    # iterate for each subject
    for sub in full_subject_list:
        # path to T1 image
        image_path = os.path.join(images_dir, sub, 'derivatives', 'registered_images', f'{sub}_hifi_T1_in_hifi_T1.nii.gz')

        success = create_SynthSeg_segmentation_single_image(sub, SynthSeg_dir, image_path, output_dir)

        if success == True:
            print(f'{sub} successful')
        
        else:
            print(f'{sub} failed')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Create SynthSeg segmentations of 3T T1 images')
    
    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # SynthSeg directory
    parser.add_argument('-synthseg_dir','--synthseg_dir',
                        help='Directory with SynthSeg installation',
                        required=True,
                        )

    # Images directory
    parser.add_argument('-images_dir','--images_dir',
                        help='Directory with 3T images',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the SynthSeg segmentations',
                        required=True,
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

        # file containing subjects
        subject_file = open(os.path.abspath(args.subs_file), 'r')

        # list of all subjects
        full_subject_list = []

        list_of_arguments = []

        # convert to list
        for line in subject_file:
            line = line.split('\n')[0]
            full_subject_list.append(line)

        if os.path.exists(os.path.abspath(args.output_dir)) == False:
            os.makedirs(os.path.abspath(args.output_dir))
        
        # create list of argument tuples
        for sub in full_subject_list:
            # path to t1 image
            image_path = os.path.join(os.path.abspath(args.images_dir), sub, 'derivatives', 'registered_images', f'{sub}_hifi_T1_in_hifi_T1.nii.gz')

            list_of_arguments.append((sub, os.path.abspath(args.synthseg_dir), image_path, os.path.abspath(args.output_dir)))
        
        # spawn up 5 processes at once
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(create_SynthSeg_segmentation_single_image, list_of_arguments)
        
        print('Finished parallel SynthSeg segmentation')
    
    # run in series
    else:
        iterate_for_all_subs(
            subs_file=os.path.abspath(args.subs_file),
            SynthSeg_dir=os.path.abspath(args.synthseg_dir),
            images_dir=os.path.abspath(args.images_dir),
            output_dir=os.path.abspath(args.output_dir)
        )
    
    print('Finished')

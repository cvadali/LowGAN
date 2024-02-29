import os
import argparse
import multiprocessing

# generate SynthSR output using coregistered 64mT T1 and T2 images
def create_SynthSR_image_single_subject(subject, SynthSR_dir, t1_path, t2_path, output_dir):
    print(f'Processing: {subject}')

    # path to script
    script = os.path.join(SynthSR_dir, 'scripts', 'predict_command_line_hyperfine.py')

    # path to output file
    output_path = os.path.join(output_dir, f'{subject}_synth_T1.nii.gz')

    # run the command
    os.system(f'python {script} {t1_path} {t2_path} {output_path}')

    success = os.path.exists(output_path)

    return success

# iterate for every subject in list
def iterate_for_all_subs(subs_file, SynthSR_dir, images_dir, output_dir):
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
    
    # iterate for each sub
    for sub in full_subject_list:
        # path to t1 image
        t1_path = os.path.join(images_dir, sub, 'derivatives', 'registered_images', f'{sub}_lofi_T1_in_hifi_T1.nii.gz')

        # path to t2 image
        t2_path = os.path.join(images_dir, sub, 'derivatives', 'registered_images', f'{sub}_lofi_T2_in_hifi_T1.nii.gz')

        success = create_SynthSR_image_single_subject(sub, SynthSR_dir, t1_path, t2_path, output_dir)

        if success == True:
            print(f'{sub} successful')
        
        else:
            print(f'{sub} failed')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Create SynthSR outputs using 64mT T1 and T2 images')
    
    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # SynthSR directory
    parser.add_argument('-synthsr_dir','--synthsr_dir',
                        help='Directory with SynthSR installation',
                        required=True,
                        )

    # Images directory
    parser.add_argument('-images_dir','--images_dir',
                        help='Directory with images',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the SynthSR images',
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

    # process in parallel, 10 at a time
    if bool(args.parallel) == True:
        max_processes = 10

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
            t1_path = os.path.join(os.path.abspath(args.images_dir), sub, 'derivatives', 'registered_images', f'{sub}_lofi_T1_in_hifi_T1.nii.gz')

            # path to t2 image
            t2_path = os.path.join(os.path.abspath(args.images_dir), sub, 'derivatives', 'registered_images', f'{sub}_lofi_T2_in_hifi_T1.nii.gz')

            list_of_arguments.append((sub, os.path.abspath(args.synthsr_dir), t1_path, t2_path, os.path.abspath(args.output_dir)))
        
        # spawn up 10 processes at once
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(create_SynthSR_image_single_subject, list_of_arguments)
        
        print('Finished parallel SynthSR processing')
    
    # run in series
    else:
        iterate_for_all_subs(
            subs_file=os.path.abspath(args.subs_file),
            SynthSR_dir=os.path.abspath(args.synthsr_dir),
            images_dir=os.path.abspath(args.images_dir),
            output_dir=os.path.abspath(args.output_dir)
        )
    
    print('Finished')



            
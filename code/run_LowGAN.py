import os
import logging
from pathlib import Path
import torch
import random
import argparse
import multiprocessing
import create_pix2pix_datasets_running_LowGAN
import test_stage1_running_LowGAN
import reconstruct_stage1_images_running_LowGAN
import reshape_stage1_images_running_LowGAN
import coregister_reshaped_reconstructed_stage1_running_LowGAN
import create_stage2_outputs_running_LowGAN


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

# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Run full LowGAN pipeline on data in BIDS format')

    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )

    # data source
    parser.add_argument('-data','--data',
                        help='Directory with subjects in BIDS format (each with T1, T2, and FLAIR images)',
                        required=True,
                        )
    
    # checkpoints directory
    parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
                        help='Directory with weights of stage 1 and stage 2 models',
                        required=True,
                        )

    # LowGAN code directory
    parser.add_argument('-LowGAN_code_dir', '--LowGAN_code_dir',
                        help='Directory with LowGAN code',
                        required=True,
                        )
    
    # pytorch-CycleGAN-and-pix2pix directory
    parser.add_argument('-pytorch_CycleGAN_and_pix2pix_dir', '--pytorch_CycleGAN_and_pix2pix_dir',
                        help='pytorch-CycleGAN-and-pix2pix directory',
                        required=True,
                        )

    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output all outputs from LowGAN',
                        required=True,
                        )
    
    # (OPTIONAL) run some sections in parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel',
                        required=False,
                        default=False,
                        )

    
    args = parser.parse_args()

    log_filename = os.path.join(os.path.abspath(args.output_dir), 'run_LowGAN.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    # print arguments
    print(args)

    print('Starting')

    logging.info('Starting')

    # list of subjects
    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    print('Number of subjects: ', len(full_subject_list))

    print('Reorienting, registering, and skullstripping')

    logging.info('Reorienting, registering, and skullstripping')

    # reorient, register, and skullstrip each image

    reorient_register_skullstrip_script = os.path.join(os.path.abspath(args.LowGAN_code_dir), 'reorient_register_skullstrip_running_LowGAN.py')

    os.system(f'python {reorient_register_skullstrip_script} --subs_file {os.path.abspath(args.subs_file)} --data {os.path.abspath(args.data)}')

    logging.info('Finished Reorienting, registering, and skullstripping')

    print('Creating pix2pix dataset')

    logging.info('Creating pix2pix dataset')

    # process in parallel
    if bool(args.parallel) == True:
        print('Creating pix2pix dataset in parallel')
        max_processes = 10

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

                output_path_test = os.path.join(os.path.abspath(args.output_dir), f'dataset_{plane}_pix2pix', 'test')

                if os.path.exists(output_path_test) == False:
                    os.makedirs(output_path_test)

                list_of_arguments.append((sub, os.path.abspath(args.data), os.path.abspath(args.output_dir), plane))
        
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(create_pix2pix_datasets_running_LowGAN.create_dataset_single_sub_single_plane, list_of_arguments)

    # run in series
    else:
        create_pix2pix_datasets_running_LowGAN.create_dataset_all_planes(
            subs_file=os.path.abspath(args.subs_file),
            data_source=os.path.abspath(args.data),
            outdir=os.path.abspath(args.output_dir)
        )
    
    print('Finished creating pix2pix dataset')

    logging.info('Finished creating pix2pix dataset')

    print('Running stage 1 of LowGAN (pix2pix network)')

    logging.info('Running stage 1 of LowGAN (pix2pix network)')

    # create stage 1 outputs
    test_stage1_running_LowGAN.test_stage1(
        model_name='LowGAN_stage1',
        data_source=os.path.abspath(args.output_dir),
        checkpoints_dir=os.path.abspath(args.checkpoints_dir),
        output_dir=os.path.abspath(args.output_dir),
        pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath(args.pytorch_CycleGAN_and_pix2pix_dir),
        direction='BtoA'
    )

    print('Finished running stage 1 of LowGAN')

    logging.info('Finished running stage 1 of LowGAN')

    print('Reconstructing images from stage 1 outputs')

    logging.info('Reconstructing images from stage 1 outputs')

    # reconstruct images from stage 1 outputs

    # process in parallel
    if bool(args.parallel) == True:
        print('Reconstructing images from stage 1 outputs in parallel')
        max_processes = 10
        
        list_of_arguments = []

        # create list of argument tuples

        output_dir = os.path.join(os.path.abspath(args.output_dir), 'reconstructed_stage1_niftis')

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        
        for plane in planes:
            # output directory for this plane
            output_dir_plane = os.path.join(output_dir, f'recon_{plane}')

            if os.path.exists(output_dir_plane) == False:
                os.makedirs(output_dir_plane)
            
            for sub in full_subject_list:
                list_of_arguments.append((sub, os.path.abspath(args.output_dir), output_dir, 'LowGAN_stage1', 
                                        plane, 'test', 'latest'))
        
        # spawn up 10 processes at once
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(reconstruct_stage1_images_running_LowGAN.save_volume_predicted_single_plane, list_of_arguments)

    # run in series
    else:
        reconstruct_stage1_images_running_LowGAN.iterate_for_each_plane(
            data_dir=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'reconstructed_stage1_niftis'),
            model_name_stem='LowGAN_stage1',
            phase='test',
            epoch='latest'
        )
    
    print('Finished reconstructing images from stage 1 outputs')

    logging.info('Finished reconstructing images from stage 1 outputs')

    print('Reshape reconstructed stage 1 images')

    logging.info('Reshape reconstructed stage 1 images')

    # reshape reconstructed stage 1 images

    # process in parallel
    if bool(args.parallel) == True:
        print('Reshape reconstructed stage 1 images in parallel')
        max_processes = 10

        list_of_arguments = []

        reconstructed_dir = os.path.join(os.path.abspath(args.output_dir), 'reconstructed_stage1_niftis')

        reshaped_dir = os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped')

        if os.path.exists(reshaped_dir) == False:
            os.makedirs(reshaped_dir)

        for plane in planes:
            for sub in full_subject_list:
                list_of_arguments.append((sub, plane, reconstructed_dir, reshaped_dir))
        
        # spawn up to 10 processes at a time
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(reshape_stage1_images_running_LowGAN.reshape_t1_t2_flair, list_of_arguments)
    
    # run in series
    else:
        reshape_stage1_images_running_LowGAN.iterate_for_each_plane(
            subject_list=full_subject_list,
            data_dir=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped')
        )
    
    print('Finished reshaping reconstructed stage 1 images')

    logging.info('Finished reshaping reconstructed stage 1 images')

    print('Coregister reshaped reconstructed stage 1 images')

    logging.info('Coregister reshaped reconstructed stage 1 images')

    # coregister reshaped reconstructed stage 1 images
    coregister_reshaped_reconstructed_stage1_running_LowGAN.coregister_all_subjects(
        subjects_list=full_subject_list,
        data_dir=os.path.abspath(args.output_dir),
        output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered')
    )

    print('Finished coregistering reshaped reconstructed stage 1 images')

    logging.info('Finished coregistering reshaped reconstructed stage 1 images')

    print('Running stage 2 of LowGAN (3D UNet)')

    logging.info('Running stage 2 of LowGAN (3D UNet)')

    # run stage 2

    try:

        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # set seed
        seed = 17  # for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)


        create_stage2_outputs_running_LowGAN.predict_all_subjects(
            subject_list=full_subject_list,
            checkpoints_dir=Path(args.checkpoints_dir),
            output_dir=Path(os.path.join(os.path.abspath(args.output_dir), 'LowGAN_stage2_outputs')),
            device=device,
            images_dir=Path(os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered')),
            batch_size=4,
            patch_size=(32,32,32),
            patch_overlap=(4,4,4)
        )

        print('Finished running stage 2 of LowGAN')

        logging.info('Finished running stage 2 of LowGAN')

    except Exception as e:
        print(e)
        logging.info(e)


    print('Finished running LowGAN')

    logging.info('Finished running LowGAN')

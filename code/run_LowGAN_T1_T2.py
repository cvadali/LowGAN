import os
import argparse
import multiprocessing
import combine_T1_T2
import generate_pix2pix_datasets_T1_T2
import LowGAN_inference
import reconstruct_volumes
import reshape_reconstructed_volumes
import coregister_reshaped_reconstructed_volumes
import wavelet_transform_filter
import unpad_T1_T2

"""
This script runs the full LowGAN pipeline on T1 and T2 data in BIDS format. The pipeline includes the following steps:

1. Synthesize contrast-enhanced T1 images from T1 and T2 images
2. Reorient, register, and skullstrip each image
3. Generate pix2pix datasets
4. Run inference with LowGAN pix2pix networks
5. Reconstruct volumes from pix2pix outputs
6. Reshape reconstructed volumes
7. Coregister reshaped reconstructed volumes
8. Filter volumes using wavelet transform
9. Unpad filtered volumes to original shape


The required arguments are:

-subs_file: Path to file containing list of subjects
-data: Path to directory with subjects in BIDS format (each with T1 and T2 images)
-output_dir: Path to directory to output all outputs from LowGAN

The optional arguments are:

-parallel: Run the pipeline in parallel
-intermediates: Keep intermediate outputs from the pipeline. This is useful for debugging

"""


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

    parser.add_argument('-subs_file','--subs_file', help='File containing list of subjects', type=str, required=True)
    parser.add_argument('-data','--data', help='Directory with subjects in BIDS format (each with T1 and T2 images)', type=str, required=True)
    parser.add_argument('-output_dir','--output_dir', help='Directory to output all outputs from LowGAN', type=str, required=True)
    parser.add_argument('-parallel','--parallel', help='(OPTIONAL) Run in parallel', action='store_true')
    parser.add_argument('-intermediates', '--intermediates', help='(OPTIONAL) Keep intermediate outputs', action='store_true')
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # path from where script is run
    # important to save because the test script changes the working directory
    original_path = os.getcwd()

    # list of subjects
    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    print('Number of subjects: ', len(full_subject_list))

    print('Synthesizing contrast-enhanced T1 images from T1 and T2 images')

    # combine T1 and T2 images
    combine_T1_T2.combine_T1_T2_all_subs(
        full_subject_list=full_subject_list,
        source_data_dir=os.path.abspath(args.data),
        mni_template_path=os.path.join(original_path, 'mni_icbm152_t1_tal_nlin_asym_09c_LPI.nii.gz'),
        gm_mask_path=os.path.join(original_path, 'gm_mask_LPI.nii.gz')
    )

    print('Registering and skullstripping')

    # register and skullstrip each image
    register_skullstrip_script = 'register_skullstrip_T1_T2.py'

    os.system(f'python {register_skullstrip_script} --subs_file {os.path.abspath(args.subs_file)} --data {os.path.abspath(args.data)}')

    print('Generating pix2pix datasets')

    parallel = args.parallel

    intermediates = args.intermediates

    # process in parallel
    if parallel:
        print('Generating pix2pix datasets in parallel')
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
            pool.starmap(generate_pix2pix_datasets_T1_T2.generate_dataset_single_sub_single_plane, list_of_arguments)

    # run in series
    else:
        generate_pix2pix_datasets_T1_T2.generate_dataset_all_planes(
            subs_file=os.path.abspath(args.subs_file),
            data_source=os.path.abspath(args.data),
            outdir=os.path.abspath(args.output_dir)
        )
    
    print('Finished generating pix2pix datasets')

    print('Running inference with LowGAN pix2pix networks')

    # run inference
    LowGAN_inference.test_model(
        model_name='LowGAN',
        data_source=os.path.abspath(args.output_dir),
        checkpoints_dir=os.path.abspath('checkpoints_T1_T2/'),
        output_dir=os.path.abspath(args.output_dir),
        pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath('pytorch-CycleGAN-and-pix2pix/'),
        direction='BtoA'
    )

    print('Finished inference with LowGAN pix2pix networks')

    # change back to original path
    os.chdir(original_path)

    print('Reconstructing images from pix2pix outputs')

    # reconstruct images from pix2pix outputs
    # process in parallel
    if parallel:
        print('Reconstructing images from pix2pix outputs in parallel')
        max_processes = 10
        
        list_of_arguments = []

        # create list of argument tuples

        output_dir = os.path.join(os.path.abspath(args.output_dir), 'reconstructed_niftis')

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        
        for plane in planes:
            # output directory for this plane
            output_dir_plane = os.path.join(output_dir, f'recon_{plane}')

            if os.path.exists(output_dir_plane) == False:
                os.makedirs(output_dir_plane)
            
            for sub in full_subject_list:
                list_of_arguments.append((sub, os.path.abspath(args.output_dir), output_dir, 'LowGAN', 
                                        plane, 'test', 'latest'))
        
        # spawn up 10 processes at once
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(reconstruct_volumes.save_volume_predicted_single_plane, list_of_arguments)

    # run in series
    else:
        reconstruct_volumes.iterate_for_each_plane(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'reconstructed_niftis'),
            model_name_stem='LowGAN',
            phase='test',
            epoch='latest'
        )
    
    print('Finished reconstructing volumes from pix2pix outputs')

    print('Reshape reconstructed volumes')

    # reshape reconstructed volumes
    
    # process in parallel
    if parallel:
        print('Reshape reconstructed volumes in parallel')
        max_processes = 10

        list_of_arguments = []

        reconstructed_dir = os.path.join(os.path.abspath(args.output_dir), 'reconstructed_niftis')

        reshaped_dir = os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped')

        if os.path.exists(reshaped_dir) == False:
            os.makedirs(reshaped_dir)

        for plane in planes:
            for sub in full_subject_list:
                list_of_arguments.append((sub, plane, reconstructed_dir, reshaped_dir))
        
        # spawn up to 10 processes at a time
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(reshape_reconstructed_volumes.reshape_t1_t2_flair, list_of_arguments)
    
    # run in series
    else:
        reshape_reconstructed_volumes.iterate_for_each_plane(
            subject_list=full_subject_list,
            data_dir=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped')
        )
    
    print('Finished reshaping reconstructed volumes')

    print('Coregister reshaped reconstructed volumes')

    # coregister reshaped reconstructed volumes
    coregister_reshaped_reconstructed_volumes.coregister_all_subjects(
        subjects_list=full_subject_list,
        data_dir=os.path.abspath(args.output_dir),
        output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered')
    )

    print('Finished coregistering reshaped reconstructed volumes')

    print('Filtering volumes using wavelet transform')

    if parallel:
        wavelet_transform_filter.filter_in_parallel(
            subjects=full_subject_list,
            path=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
            decNum=8,
            wname='db25',
            damp_sigma=8
        )
    
    else:
        wavelet_transform_filter.filter_in_series(
            subjects=full_subject_list,
            path=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
            decNum=8,
            wname='db25',
            damp_sigma=8
        )

    print('Finished filtering volumes using wavelet transform')

    print('Coregistering filtered outputs')

    print('Finished coregistering filtered outputs')

    print('Unpadding volumes to original shape')

    if parallel:
        unpad_T1_T2.unpad_parallel(
            subjects=full_subject_list,
            padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
            orig_dir=os.path.abspath(args.data),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
        )
    
    else:
        unpad_T1_T2.unpad_series(
            subjects=full_subject_list,
            padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
            orig_dir=os.path.abspath(args.data),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
        )

    print('Finished unpadding volumes to original shape')

    if intermediates:
        print('Moving intermediate outputs to intermediates directory')

        if os.path.exists(os.path.join(os.path.abspath(args.output_dir), 'intermediates')) == False:
            os.makedirs(os.path.join(os.path.abspath(args.output_dir), 'intermediates'))

        # we will move only the specific directories that were generated by LowGAN to the intermediates directory
        # to make sure that we don't mess with any other files that may be in the output directory

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        
        os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "results_LowGAN")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
        os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "reconstructed_niftis")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
        os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
        os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
        os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered_filtered")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        print('Finished moving intermediate outputs to intermediates directory')

    else:
        print('Removing intermediate outputs')

        # we will move only the specific directories that were generated by LowGAN to the intermediates directory
        # to make sure that we don't mess with any other files that may be in the output directory

        # remove generated files from reorient_register_skullstrip.py
        for sub in full_subject_list:
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_combined_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_combined_to_lofi_t1_xfm.mat")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T1_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T1_to_lofi_t1_xfm.mat")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T2_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T2_to_lofi_t1_xfm.mat")}')

            os.system(f'rm {os.path.join(os.path.abspath(args.output_dir), "LowGAN_outputs", sub + "_LowGAN_FLAIR.nii.gz")}')

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")}')

        os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "results_LowGAN")}')
        os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "reconstructed_niftis")}')
        os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped")}')
        os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered")}')
        os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered_filtered")}')

        print('Finished removing intermediate outputs')


    print('Finished running LowGAN')

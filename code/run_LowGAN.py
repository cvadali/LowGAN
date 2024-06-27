import os
import argparse
import multiprocessing
import generate_pix2pix_datasets
import LowGAN_inference
import LowGAN_inference_ensemble
import reconstruct_volumes
import reconstruct_volumes_ensemble
import reshape_reconstructed_volumes
import reshape_reconstructed_volumes_ensemble
import coregister_reshaped_reconstructed_volumes
import coregister_reshaped_reconstructed_volumes_ensemble
import wavelet_transform_filter
import wavelet_transform_filter_ensemble
import coregister_filtered_outputs_ensemble
import averaging_filtered_ensemble
import unpad

"""
This script runs the full LowGAN pipeline on data in BIDS format. The pipeline includes the following steps:

1. Reorient, register, and skullstrip each image
2. Generate pix2pix datasets
3. Run inference with LowGAN pix2pix networks
4. Reconstruct volumes from pix2pix outputs
5. Reshape reconstructed volumes
6. Coregister reshaped reconstructed volumes
7. Filter volumes using wavelet transform
8. Unpad filtered volumes to original shape


The required arguments are:

-subs_file: Path to file containing list of subjects
-data: Path to directory with subjects in BIDS format (each with T1, T2, and FLAIR images)
-output_dir: Path to directory to output all outputs from LowGAN

The optional arguments are:

-parallel: Run the pipeline in parallel
-ensemble: Use an ensemble of 12 models and average the outputs. This generates more robust results but takes longer
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
    parser.add_argument('-data','--data', help='Directory with subjects in BIDS format (each with T1, T2, and FLAIR images)', type=str, required=True)
    parser.add_argument('-output_dir','--output_dir', help='Directory to output all outputs from LowGAN', type=str, required=True)
    parser.add_argument('-parallel','--parallel', help='(OPTIONAL) Run in parallel', action='store_true')
    parser.add_argument('-ensemble', '--ensemble', help='(OPTIONAL) Use ensemble of 12 models and average', action='store_true')
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

    print('Reorienting, registering, and skullstripping')

    # reorient, register, and skullstrip each image
    reorient_register_skullstrip_script = 'reorient_register_skullstrip.py'

    os.system(f'python {reorient_register_skullstrip_script} --subs_file {os.path.abspath(args.subs_file)} --data {os.path.abspath(args.data)}')

    print('Generating pix2pix datasets')

    parallel = args.parallel

    ensemble = args.ensemble

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
            pool.starmap(generate_pix2pix_datasets.generate_dataset_single_sub_single_plane, list_of_arguments)

    # run in series
    else:
        generate_pix2pix_datasets.generate_dataset_all_planes(
            subs_file=os.path.abspath(args.subs_file),
            data_source=os.path.abspath(args.data),
            outdir=os.path.abspath(args.output_dir)
        )
    
    print('Finished generating pix2pix datasets')

    print('Running inference with LowGAN pix2pix networks')

    # run inference

    if ensemble:
        print('Ensemble')

        LowGAN_inference_ensemble.test_all_folds(
            model_dir=os.path.abspath('checkpoints/LowGAN_ensemble/'),
            model_name='LowGAN_ensemble',
            data_source=os.path.abspath(args.output_dir),
            output_dir=os.path.abspath(args.output_dir),
            pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath('pytorch-CycleGAN-and-pix2pix/'),
            n_splits=12
        )

    else:
        print('Normal')

        LowGAN_inference.test_model(
            model_name='LowGAN',
            data_source=os.path.abspath(args.output_dir),
            checkpoints_dir=os.path.abspath('checkpoints/'),
            output_dir=os.path.abspath(args.output_dir),
            pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath('pytorch-CycleGAN-and-pix2pix/'),
            direction='BtoA'
        )

    print('Finished inference with LowGAN pix2pix networks')

    # change back to original path
    os.chdir(original_path)

    print('Reconstructing images from pix2pix outputs')

    # reconstruct images from pix2pix outputs

    if ensemble:
        print('Ensemble')
        # process in parallel
        if parallel:
            print('Reconstructing images from pix2pix outputs in parallel')
            max_processes = 10
            
            list_of_arguments = []

            # get list of folds
            folds = [f'fold_{fold}' for fold in range(0, 12)]

            # create list of argument tuples
                
            for fold in folds:
                list_of_arguments.append((fold, os.path.abspath(args.output_dir),
                                        os.path.abspath(args.output_dir),
                                        'LowGAN_ensemble', 'test', 'latest'))
            
            # spawn up 10 processes at once
            with multiprocessing.Pool(processes=max_processes) as pool:
                pool.starmap(reconstruct_volumes_ensemble.iterate_for_each_plane, list_of_arguments)

        else:
            reconstruct_volumes_ensemble.iterate_for_each_fold(
                data_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                model_name_stem='LowGAN_ensemble',
                phase='test',
                epoch='latest',
                n_splits=12
            )

    else:
        print('Normal')
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
                data_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'reconstructed_niftis'),
                model_name_stem='LowGAN',
                phase='test',
                epoch='latest'
            )
    
    print('Finished reconstructing volumes from pix2pix outputs')

    print('Reshape reconstructed volumes')

    # reshape reconstructed volumes

    if ensemble:
        print('Ensemble')

        if parallel:
            reshape_reconstructed_volumes_ensemble.iterate_in_parallel(
                subs_file=os.path.abspath(args.subs_file),
                data_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                n_splits=12
            )
        
        else:
            reshape_reconstructed_volumes_ensemble.iterate_for_each_fold(
                subs_file=os.path.abspath(args.subs_file),
                data_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                n_splits=12
            )


    else:
        print('Normal')

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
    if ensemble:
        print('Ensemble')

        coregister_reshaped_reconstructed_volumes_ensemble.coregister_all_folds(
            subs_file=os.path.abspath(args.subs_file),
            data_dir=os.path.abspath(args.output_dir),
            out_dir=os.path.abspath(args.output_dir),
            n_splits=12
        )

    else:
        print('Normal')

        coregister_reshaped_reconstructed_volumes.coregister_all_subjects(
            subjects_list=full_subject_list,
            data_dir=os.path.abspath(args.output_dir),
            output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered')
        )

    print('Finished coregistering reshaped reconstructed volumes')

    print('Filtering volumes using wavelet transform')

    if ensemble:
        if parallel:
            print('Ensemble')

            wavelet_transform_filter_ensemble.filter_in_parallel(
                subs_file=os.path.abspath(args.subs_file),
                path=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                n_splits=12,
                decNum=8,
                wname='db25',
                damp_sigma=8
            )
        
        else:
            wavelet_transform_filter_ensemble.filter_in_series(
                subs_file=os.path.abspath(args.subs_file),
                path=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                n_splits=12,
                decNum=8,
                wname='db25',
                damp_sigma=8
            )

    else:
        print('Normal')

        if parallel:
            wavelet_transform_filter.filter_in_parallel(
                path=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
                decNum=8,
                wname='db25',
                damp_sigma=8
            )
        
        else:
            wavelet_transform_filter.filter_in_series(
                path=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
                decNum=8,
                wname='db25',
                damp_sigma=8
            )

    print('Finished filtering volumes using wavelet transform')

    print('Coregistering filtered outputs')

    if ensemble:
        coregister_filtered_outputs_ensemble.coregister_all_subjects(
            subs_file=os.path.abspath(args.subs_file),
            ensemble_output_dir=os.path.abspath(args.output_dir),
            n_splits=12
        )

    print('Finished coregistering filtered outputs')

    # average the outputs across each fold to increase signal-to-noise ratio
    if ensemble:
        print('Averaging filtered volumes')

        if parallel:
            print('Averaging in parallel')

            averaging_filtered_ensemble.iterate_in_parallel(
                subs_file=os.path.abspath(args.subs_file),
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_filtered_averaged_across_folds'),
                n_splits=12
            )

        else:
            print('Averaging in series')

            averaging_filtered_ensemble.iterate_for_each_sub(
                subs_file=os.path.abspath(args.subs_file),
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_filtered_averaged_across_folds'),
                n_splits=12
            )

        print('Finished averaging filtered volumes')

    print('Unpadding volumes to original shape')

    if ensemble:
        if parallel:
            unpad.unpad_parallel(
                subjects=full_subject_list,
                padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_filtered_averaged_across_folds'),
                orig_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
            )
        
        else:
            unpad.unpad_series(
                subjects=full_subject_list,
                padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_filtered_averaged_across_folds'),
                orig_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
            )

    else:
        if parallel:
            unpad.unpad_parallel(
                subjects=full_subject_list,
                padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
                orig_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
            )
        
        else:
            unpad.unpad_series(
                subjects=full_subject_list,
                padded_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_reshaped_coregistered_filtered'),
                orig_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'LowGAN_outputs')
            )

    print('Finished unpadding volumes to original shape')

    if intermediates:
        print('Moving intermediate outputs to intermediates directory')

        os.makedirs(os.path.join(os.path.abspath(args.output_dir), 'intermediates'))

        # we will move only the specific directories that were generated by LowGAN to the intermediates directory
        # to make sure that we don't mess with any other files that may be in the output directory

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        if ensemble:
            folds = [f'fold_{fold}' for fold in range(0, 12)]

            for fold in folds:
                os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), fold)} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
            
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_filtered_averaged_across_folds")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        else:
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
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_brainmask.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_FLAIR_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_FLAIR_in_LPI.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_FLAIR_to_lofi_t1_xfm.mat")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T1_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T1_in_LPI.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T1_to_lofi_t1_xfm.mat")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T2_in_lofi_T1.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T2_in_LPI.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_lofi_T2_to_lofi_t1_xfm.mat")}')

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")}')

        if ensemble:
            folds = [f'fold_{fold}' for fold in range(0, 12)]

            for fold in folds:
                os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), fold)}')
            
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_filtered_averaged_across_folds")}')

        else:
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "results_LowGAN")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "reconstructed_niftis")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_reshaped_coregistered_filtered")}')

        print('Finished removing intermediate outputs')


    print('Finished running LowGAN')

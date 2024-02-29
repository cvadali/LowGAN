import os
import argparse
import sys
import multiprocessing

# test stage 1 network in a single plane
def test_single_plane(model_name, dataset_name, direction, checkpoints_dir, results_dir):
    # train model in a single plane
    command = f'python /mnt/leif/littlab/users/cvadali/LowGAN_leave_one_out/code/pytorch-CycleGAN-and-pix2pix/test.py --dataroot {dataset_name} ' \
        + f'--name {model_name} --model pix2pix --direction {direction} --preprocess resize_and_crop ' \
        + f'--num_test 100000 --epoch latest --phase train --checkpoints_dir {checkpoints_dir} --results_dir {results_dir}'
    
    print(command)
    
    os.system(command)

    print('Testing on training set complete')

    if os.path.exists(os.path.join(results_dir, model_name)) == True:
        print(f'Successfully created {dataset_name} training set outputs')
    
    else:
        print(f'Failed to create {dataset_name} training set outputs')

# test stage 1 of the model for a single fold
def test_stage1(model_name, fold, data_dir, output_dir, direction='BtoA'):

    data_source = os.path.join(data_dir, fold)

    output_dir_fold = os.path.join(output_dir, fold)

    if os.path.exists(output_dir_fold) == False:
        os.makedirs(output_dir_fold)
    
    os.chdir(output_dir_fold)

    # three planes
    planes = ['axial','coronal','sagittal']

    for plane in planes:
        # set model name
        name_of_model = f'{model_name}_{plane}'

        # set dataset name
        dataset_name = os.path.join(data_source, f'dataset_{plane}_pix2pix')

        # checkpoints dir
        checkpoints_dir = os.path.join(output_dir_fold, 'checkpoints')

        # results dir
        results_dir = os.path.join(output_dir_fold, f'results_stage1_training_set')

        print(model_name)
        print(dataset_name)
        print(plane)
        print(checkpoints_dir)
        print(results_dir)

        # train model in plane
        test_single_plane(name_of_model, dataset_name, direction, checkpoints_dir, results_dir)

# loop for each fold
def test_stage1_all_folds(model_name, data_source, output_dir, direction='BtoA', n_splits=12):

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    print(f'Folds: {folds}')

    for fold in folds:
        try:
            print(f'Testing stage 1 training set: {fold}')
        
            test_stage1(f'{model_name}_{fold}', fold, data_source, output_dir, direction)

            print(f'Finished testing stage 1 training set: {fold}')

        except Exception as e:
            print(e)
            sys.exit()


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Test Stage 1 of LowGAN on training set')

    # model name
    parser.add_argument('-model_name','--model_name',
                        help='Name of the model',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with data in pix2pix format',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory where the model outputs are',
                        required=True,
                        )
    
    # (OPTIONAL) direction in which to train network
    parser.add_argument('-direction','--direction',
                        help='Direction in which to test the model',
                        default='BtoA',
                        required=False,
                        )

    # (OPTIONAL) Number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of folds',
                        default=12,
                        required=False,
                        )
    
    # (OPTIONAL) parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel, 12 at a time',
                        default=False,
                        required=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # run in parallel
    if bool(args.parallel) == True:
        max_processes = 12

        # get list of folds
        folds = [f'fold_{fold}' for fold in range(0, int(args.n_splits))]

        print(f'Folds: {folds}')
        
        list_of_arguments = []

        # create list of argument tuples
        for fold in folds:
            name_of_the_model = f'{args.model_name}_{fold}'
            list_of_arguments.append((name_of_the_model, fold, args.data, args.output_dir, args.direction))
        
        # spawn up to 12 processes at a time
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(test_stage1, list_of_arguments)

        print('Finished parallel testing')

    # run in series
    else:
        # train stage 1
        test_stage1_all_folds(
            model_name=args.model_name,
            data_source=args.data,
            output_dir=args.output_dir,
            direction=args.direction,
            n_splits=int(args.n_splits)
        )

import os
import multiprocessing
import argparse
import LowGAN_inference

def test_single_fold(fold, model_dir, model_name, data_source, output_dir, pytorch_CycleGAN_and_pix2pix_dir, direction='BtoA'):
    # set model name
    name_of_model = f'{model_name}_{fold}'

    checkpoints_dir_fold = os.path.join(model_dir, fold)

    # output directory
    output_dir_fold = os.path.join(output_dir, fold)

    LowGAN_inference.test_model(
        model_name=name_of_model,
        data_source=data_source,
        checkpoints_dir=checkpoints_dir_fold,
        output_dir=output_dir_fold,
        pytorch_CycleGAN_and_pix2pix_dir=pytorch_CycleGAN_and_pix2pix_dir,
        direction=direction
    )

def test_all_folds(model_dir, model_name, data_source, output_dir, pytorch_CycleGAN_and_pix2pix_dir, direction='BtoA', n_splits=12):
    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, int(n_splits))]

    if os.path.exists(os.path.abspath(output_dir)) == False:
        os.makedirs(os.path.abspath(output_dir))

    for fold in folds:
        test_single_fold(
            fold=fold,
            model_dir=model_dir,
            model_name=model_name,
            data_source=data_source,
            output_dir=output_dir,
            pytorch_CycleGAN_and_pix2pix_dir=pytorch_CycleGAN_and_pix2pix_dir,
            direction=direction
        )

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Run Inference using 12 LowGAN models in an ensemble method')

    # model directory
    parser.add_argument('-model_dir','--model_dir',
                        help='Directory where output models from all folds are',
                        required=True,
                        )

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
                        help='Directory to output results',
                        required=True,
                        )
    
    # pytorch-CycleGAN-and-pix2pix directory
    parser.add_argument('-pytorch_CycleGAN_and_pix2pix_dir', '--pytorch_CycleGAN_and_pix2pix_dir',
                        help='pytorch-CycleGAN-and-pix2pix directory',
                        required=True,
                        )
    
    # (OPTIONAL) direction in which to test network
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
    
    # (OPTIONAL) test in parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel',
                        default=False,
                        required=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # process in parallel
    if bool(args.parallel) == True:
        print('Create outputs in parallel')
        max_processes = int(args.n_splits)

        if os.path.exists(os.path.abspath(args.output_dir)) == False:
            os.makedirs(os.path.abspath(args.output_dir))

        # get list of folds
        folds = [f'fold_{fold}' for fold in range(0, int(args.n_splits))]

        list_of_args = []

        for fold in folds:
            list_of_args.append((fold, 
                                 os.path.abspath(args.model_dir), 
                                 args.model_name, 
                                 os.path.abspath(args.data), 
                                 os.path.abspath(args.output_dir), 
                                 os.path.abspath(args.pytorch_CycleGAN_and_pix2pix_dir), 
                                 args.direction))
            
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(test_single_fold, list_of_args)
        
        print('Finished testing in parallel')

    
    # run in series
    else:
        test_all_folds(
            model_dir=os.path.abspath(args.model_dir),
            model_name=args.model_name,
            data_source=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath(args.pytorch_CycleGAN_and_pix2pix_dir),
            direction=args.direction,
            n_splits=int(args.n_splits)
        )

    print('Finished')
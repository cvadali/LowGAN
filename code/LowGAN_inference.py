import os
import argparse
import sys
import torch
import multiprocessing

# test network in a single plane
def test_single_plane(model_name, dataset_name, direction, checkpoints_dir, results_dir, pytorch_CycleGAN_and_pix2pix_dir):
    test_script = os.path.join(pytorch_CycleGAN_and_pix2pix_dir, 'test.py')

    if torch.cuda.is_available() == True:
        gpu_ids = '0'

    else:
        gpu_ids = '-1'

    # test model in a single plane
    command = f'python {test_script} --dataroot {dataset_name} ' \
        + f'--name {model_name} --model pix2pix --direction {direction} --preprocess resize_and_crop ' \
        + f'--num_test 100000 --epoch latest --phase test --checkpoints_dir {checkpoints_dir} --results_dir {results_dir} ' \
        + f'--gpu_ids {gpu_ids}'
    
    print(command)
    
    os.system(command)

    print('Testing complete')

    if os.path.exists(os.path.join(results_dir, model_name)) == True:
        print(f'Successfully created {dataset_name} test set outputs')
    
    else:
        print(f'Failed to create {dataset_name} test set outputs')

# test model
def test_model(model_name, data_source, checkpoints_dir, output_dir, pytorch_CycleGAN_and_pix2pix_dir, direction='BtoA'):

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    os.chdir(output_dir)

    # three planes
    planes = ['axial','coronal','sagittal']

    for plane in planes:
        # set model name
        name_of_model = f'{model_name}_{plane}'

        # set dataset name
        dataset_name = os.path.join(data_source, f'dataset_{plane}_pix2pix')

        # results dir
        results_dir = os.path.join(output_dir, f'results_LowGAN')

        print(model_name)
        print(dataset_name)
        print(plane)
        print(checkpoints_dir)
        print(results_dir)

        # test model in plane
        test_single_plane(name_of_model, dataset_name, direction, checkpoints_dir, results_dir, pytorch_CycleGAN_and_pix2pix_dir)


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='LowGAN Inference')

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

    # checkpoints directory
    parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
                        help='Directory with weights of model',
                        required=True,
                        )
    
    # pytorch-CycleGAN-and-pix2pix directory
    parser.add_argument('-pytorch_CycleGAN_and_pix2pix_dir', '--pytorch_CycleGAN_and_pix2pix_dir',
                        help='pytorch-CycleGAN-and-pix2pix directory',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory where the model outputs are',
                        required=True,
                        )
    
    # (OPTIONAL) direction in which to test network
    parser.add_argument('-direction','--direction',
                        help='Direction in which to test the model',
                        default='BtoA',
                        required=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # test model
    test_model(
        model_name=args.model_name,
        data_source=args.data,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        pytorch_CycleGAN_and_pix2pix_dir=args.pytorch_CycleGAN_and_pix2pix_dir,
        direction=args.direction
    )

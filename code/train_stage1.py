import numpy as np
import os
import sys
import argparse


# train stage 1 in a single plane
def train_single_plane(model_name, dataset_name, direction):
    # train model in a single plane
    command = f'python /mnt/leif/littlab/users/cvadali/LowGAN_leave_one_out/code/pytorch-CycleGAN-and-pix2pix/train.py ' \
        + f'--dataroot {dataset_name} --name {model_name} --model pix2pix --direction {direction} ' \
        + '--preprocess resize_and_crop --save_epoch_freq 100 --display_id -1 --lr 0.0001 --n_epochs 30 ' \
        + '--n_epochs_decay 70 --netG unet_256 --phase train --batch_size 2 --gpu_ids 0'
    
    print(command)
    
    os.system(command)

    print('Training complete')

# train stage 1 of the model
def train_stage1(model_name, data_source, output_dir, direction='BtoA'):
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

        print(model_name)
        print(dataset_name)
        print(plane)

        # train model in plane
        train_single_plane(name_of_model, dataset_name, direction)

        

# loop for each fold
def train_stage1_all_folds(model_name, data_source, output_dir, direction='BtoA', n_splits=12):

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    print(f'Folds: {folds}')

    for fold in folds:
        if fold == 'fold_0':
            continue
        elif fold == 'fold_1':
            continue
        elif fold == 'fold_2':
            continue
        elif fold == 'fold_3':
            continue
        else:
            try:
                print(f'Training stage 1: {fold}')
                output_dir_fold = os.path.join(output_dir, fold)

                if os.path.exists(output_dir_fold) == False:
                    os.makedirs(output_dir_fold)
            
                train_stage1(f'{model_name}_{fold}', os.path.join(data_source, fold), output_dir_fold, direction)

                print(f'Finished training stage 1: {fold}')

            except Exception as e:
                print(e)
                sys.exit()

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Train Stage 1 of LowGAN for each fold in k folds')

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
                        help='Directory to output the model',
                        required=True,
                        )
    
    # (OPTIONAL) direction in which to train network
    parser.add_argument('-direction','--direction',
                        help='Direction in which to train the model',
                        default='BtoA',
                        required=False,
                        )

    # (OPTIONAL) number of folds
    parser.add_argument('-n_splits', '--n_splits',
                        help='Number of folds',
                        required=False,
                        default=12,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # train stage 1
    train_stage1_all_folds(
        model_name=args.model_name,
        data_source=args.data,
        output_dir=args.output_dir,
        direction=args.direction,
        n_splits=int(args.n_splits)
    )
    
    print('Finished')



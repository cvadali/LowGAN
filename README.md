# LowGAN

This repository contains our implementation of LowGAN — a generative adversarial network (GAN) for low-to-high field (64mT to 3T) MR image translation. This network was trained on paired multiple sclerosis (MS) data, but feel free to try it out on any 64mT images you may have! This network takes **64mT T1w, T2w, and FLAIR sequences as input, and returns synthetic 3T T1w, T2w, and FLAIR outputs**. 

The code can run on either GPU or CPU, and provides the option of running certain steps of the pipeline in parallel or series, and also whether to use the standard model trained on 50 subjects or to use the "ensemble" model and average the outputs at the end. The ensemble model was created because we performed 12-fold cross-validation on our data, meaning we split the 50 subjects into 12 different folds, where each subject was part of the training set in 11 of the folds and part of the test set in exactly one fold, and then trained and tested a separate version of our network for each of these 12 folds. As a result, we ended up with 12 trained LowGAN networks, and we found that while every model produces an output with a little noise, different models don't necessarily produce the exact same noise for the same input (i.e., the noise is a bit random); therefore, if we use each of the 12 models to create a set of outputs and then average those outputs across the 12 models, we get **outputs with improved signal-to-noise ratios**. The tradeoff is that the ensemble model takes longer to run and is more computationally expensive, as it involves computing the outputs of 12 LowGAN networks instead of just 1, so it may not be the best choice for every use case. The normal model already performs quite well, but for particularly challenging low-field inputs, the ensemble model may be a better approach (albeit a more length and computationally expensive one).

## Installation

1. Clone this GitHub repository.
2. Create a virtual conda environment to install the necessary packages using the chosen requirements file by running the following command:
    `conda create -n LowGAN python=3.11`

    Then, activate the conda environment with the following command:
    `conda activate LowGAN`

    Then, if you want to run the code on **CPU**, run the following command:
    `pip install -r requirements_cpu.txt`

    If you instead want to run the code on **GPU**, run the following command:
    `pip install -r requirements_gpu.txt`

3. To clone the necessary pix2pix submodule, run `git submodule init` and then `git submodule update`
4. Download the `checkpoints.tar.gz` file containing the models from this [Google Drive](https://drive.google.com/file/d/14mVt9EzaBUEqHAQ3eQJgbndR9GSYTVRY/view?usp=sharing) and then copy the file to the [code](https://github.com/cvadali/LowGAN/tree/main/code) directory. 
    
    Unarchive and unzip the file by running the following command: `tar -xvzf checkpoints.tar.gz`

    After it has been unarchived and unzipped, you can remove the `checkpoints.tar.gz` file using `rm checkpoints.tar.gz`

## Data format

Your data should be in the following format:

```
data
├── P001
│   └── session1_64mT
│       └── anat
│           ├── P001_T1.nii.gz
│           ├── P001_T2.nii.gz
│           └── P001_FLAIR.nii.gz
└── P002
    └── session1_64mT
        └── anat
            ├── P002_T1.nii.gz
            ├── P002_T2.nii.gz
            └── P002_FLAIR.nii.gz
```

You will also need to **create a .txt file with the ID of each subject**. For example, above, there are 2 subjects, P001 and P002, so the subject file, which I will call `list_of_subjects.txt`, will look like this:
```
P001
P002
```

If there were 20 subjects, the file would look like this:
```
P001
P002
P003
.
.
.

P020
```

Just make sure to **place each subject ID on a new line**


## Usage

Make sure that **you are in the `code` directory**. If you are not there, run `cd code` from the `LowGAN` directory to get there.

Additionally, **make sure your data is in the BIDS format specified above and that you created a .txt file with each subject**.

Now, you can run the code with just one command:

`python run_LowGAN.py --subs_file <subs_file> --data <data> --output_dir <output_dir> [--parallel --ensemble --intermediates --skullstripped]`

**Necessary arguments**:

- `<subs_file>` is the path to a .txt file in the style above that contains the IDs of the subjects that you will be running
- `<data>` is the path to a directory containing the data in the format above
- `<output_dir>` is the path of the directory where you want to save the outputs

**Optional arguments**:

- `<parallel>` runs the creation of the pix2pix datasets, reconstruction of volumes from pix2pix outputs, reshaping of reconstructed volumes, and filtering of volumes using wavelet transform steps in parallel instead of in series (but it is computationally more expensive)
- `<ensemble>` creates outputs using 12 LowGAN models instead of just 1 and then averages the final outputs to improve signal-to-noise ratio. This approach takes longer and is more computationally expensive, but it can be useful if the low-field inputs are particularly challenging
- `<intermediates>` keeps the intermediate files generated by the pipeline and puts them in `<output_dir>/intermediates`. Normally, the intermediate files are removed, as they can take up a decent amount of space, but this can be useful for debugging
- `<skullstripped>` is important if your inputs are already skullstripped (brain only). Attempting to skullstrip a volume that is already skullstripped can mess up the outputs, so use this flag if your inputs are already skullstripped


Congrats! Your final outputs should be in `<output_dir>/LowGAN_outputs`


### Running LowGAN with T1 and T2 only (no FLAIR)

We understand that some people only have T1 and T2 sequences and not FLAIR, so we also trained a version of LowGAN to use only T1 and T2. Here's how to use it:

Download the `checkpoints_T1_T2.tar.gz` file containing the models from this [Google Drive](https://drive.google.com/file/d/1NumVsv60IgtPOYZ-_RDcHRPBeGREsrr2/view?usp=sharing) and then copy the file to the [code](https://github.com/cvadali/LowGAN/tree/main/code) directory. 
    
Unarchive and unzip the file by running the following command: `tar -xvzf checkpoints_T1_T2.tar.gz`

After it has been unarchived and unzipped, you can remove the `checkpoints_T1_T2.tar.gz` file using `rm checkpoints_T1_T2.tar.gz`

Make sure that **you are in the `code` directory**. If you are not there, run `cd code` from the `LowGAN` directory to get there.

Additionally, **make sure your data is in the BIDS format specified above and that you created a .txt file with each subject** (just with no FLAIR).

Now, you can run the code with just one command:

`python run_LowGAN_T1_T2.py --subs_file <subs_file> --data <data> --output_dir <output_dir> [--parallel --intermediates]`

**Necessary arguments**:

- `<subs_file>` is the path to a .txt file in the style above that contains the IDs of the subjects that you will be running
- `<data>` is the path to a directory containing the data in the format above
- `<output_dir>` is the path of the directory where you want to save the outputs

**Optional arguments**:

- `<parallel>` runs the creation of the pix2pix datasets, reconstruction of volumes from pix2pix outputs, reshaping of reconstructed volumes, and filtering of volumes using wavelet transform steps in parallel instead of in series (but it is computationally more expensive)
- `<intermediates>` keeps the intermediate files generated by the pipeline and puts them in `<output_dir>/intermediates`. Normally, the intermediate files are removed, as they can take up a decent amount of space, but this can be useful for debugging


Congrats! Your final outputs should be in `<output_dir>/LowGAN_outputs`

## Example

We have provided an example MS subject in the `sample_data` directory. Here, the `sample_data` directory is in the BIDS format specified above, which is necessary for the pipeline to work. We have also have provided a corresponding file, `sample_list_of_subjects.txt`, which lists the sample subject, in the format described above.

To try out this sample data, you can run the following command:

`python run_LowGAN.py --subs_file ../sample_list_of_subjects.txt --data ../sample_data/ --output_dir ../sample_data_outputs/`

This command will run LowGAN on the example subject and save the outputs in `sample_data_outputs`. The final LowGAN outputs will be in `sample_data_outputs/LowGAN_outputs`

Now that you have seen how it works, feel free to try it out with your data!


## Citation

**Multisequence 3-T Image Synthesis from 64-mT Low-Field-Strength MRI Using Generative Adversarial Networks in Multiple Sclerosis**

Lucas A, Arnold TC, Okar SV, Vadali C, Kawatra KD, Ren Z, Cao Q, Shinohara RT, Schindler MK, Davis KA, Litt B, Reich DS, Stein JM.

_Radiology_ (2025)

[Link to paper](https://pubs.rsna.org/doi/10.1148/radiol.233529)

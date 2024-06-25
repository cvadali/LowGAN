# LowGAN

This repository contains our implementation of LowGAN — a generative adversarial network (GAN) for low-to-high field (64mT to 3T) MR image translation. This network was trained on paired multiple sclerosis data, but feel free to try it out on any 64mT images you may have! This network takes **64mT T1w, T2w, and FLAIR sequences as input, and returns synthetic 3T T1w, T2w, and FLAIR outputs**. 

The code can run on either GPU or CPU, and provides the option of running certain steps of the pipeline in parallel or series, and also whether to use the standard model trained on 50 subjects or to use the "ensemble" model and average the outputs at the end. The ensemble model was created because we performed 12-fold cross-validation on our data, meaning we split the 50 subjects into 12 different folds, where each subject was part of the training set in 11 of the folds and part of the test set in exactly one fold, and then trained and tested a separate version of our network for each of these 12 folds. As a result, we ended up with 12 trained LowGAN networks, and we found that while every model produces an output with a little noise, different models don't necessarily produce the exact same noise for the same input (i.e., the noise is a bit random); therefore, if we use each of the 12 models to create a set of outputs and then average those outputs across the 12 models, we get **outputs with improved signal-to-noise ratios**. The tradeoff is that the ensemble model takes longer to run and is more computationally expensive, as it involves computing the outputs of 12 LowGAN networks instead of just 1, so it may not be the best choice for every use case. The normal model already performs quite well, but for particularly challenging low-field inputs, the ensemble model may be a better approach (albeit a more length and computationally expensive one).

## Installation

1. Clone this GitHub repository.
2. Create a virtual environment (conda or pip) to install the necessary packages using the `requirements.txt` file:
    `pip install -r requirements.txt`
    `conda create --name LowGAN --file requirements.txt`

3. Download the `checkpoints.tar.gz` file containing the models from this [Google Drive](https://drive.google.com/file/d/1pwL7TSEp0Ve-9m3o-XWx59tlz9M97uY7/view?usp=drive_link) and then copy the file to [code](https://github.com/cvadali/LowGAN/tree/main/code). Unarchive and unzip the file by running the following command: `tar -xvzf checkpoints.tar.gz`
4. If you installed the necessary packages using a conda environment, activate the environment using `conda activate LowGAN`

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

`python run_LowGAN.py --subs_file <subs_file> --data <data> --output_dir <output_dir> [--parallel --ensemble]`

**Necessary arguments**:

- `<subs_file>` is the path to a .txt file in the style above that contains the IDs of the subjects that you will be running
- `<data>` is the path to a directory containing the data in the format above
- `<output_dir>` is the path of the directory where you want to save the outputs

**Optional arguments**:

- `<parallel>` runs the creation of the pix2pix datasets, reconstruction of volumes from pix2pix outputs, reshaping of reconstructed volumes, and filtering of volumes using wavelet transform steps in parallel instead of in series (but it is computationally more expensive)
- `<ensemble>` creates outputs using 12 LowGAN models instead of just 1 and then averages the final outputs to improve signal-to-noise ratio. This approach takes longer and is more computationally expensive, but it can be useful if the low-field inputs are particularly challenging


Congrats! Your final outputs should be in the **`LowGAN_outputs`** directory within `<output_dir>`


## Citation

**Multi-contrast high-field quality image synthesis for portable low-field MRI using generative adversarial networks and paired data in multiple sclerosis**

Lucas A, Arnold TC, Okar SV, Vadali C, Kawatra KD, Ren Z, Cao Q, Shinohara RT, Schindler MK, Davis KA, Litt B, Reich DS, Stein JM.

_Under Review_ (2024)

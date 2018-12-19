# Mask R-CNN for brain slices segmentation
This repository contains all the code that was needed to run mask rcnn algorithm over the training data of brain/tissue slices as precised in the report.
## Requirements
Please be sure to have all the following packages installed and working:
- tensorflow
- keras
- skimage
If you have the tensorflow-gpu implementation, be sure to have everything ready following this guide:
https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10
To perform the training, you would need a powerful enough gpu. We used
paperspace, setting it up following this guide:
https://github.com/BrainSegmentation/lipschitz-lizards/wiki/Paperspace-Setup
which was kindly provided by another group working with us.

For the visualization part, we have made a slight modification to the visualization function in Mask_rcnn. Clone Mask RCNN from here:
https://github.com/matterport/Mask_RCNN/
and copy our visualize.py in their mrcnn folder. This modification is needed for single patch validation but not for the whole image results. It's used to have 2 channels visualization which is normally not supported. After modifying the file, you can install mask rcnn in your machine (python setup.py install in mask rcnn folder).

## Creating the dataset

To create the dataset you will need to download the original data and put it in a new /data folder. create_database.py script will patch the images contained in data.
On the computer we used (ASUS X756UX) it took approximately 6 hours. Moreover, you should manually move the result of the subdivision in two
directories, database/train and database/validate. You should respect the
structure created by create_database.py, i.e. patch j from image i for training should be in path database/train/image_i/patch_j. If you don't want to follow the process, we provide a download link to the database.

You will need the database to observe the results of single patch validation, but not to see the final (whole image) results.

## training

For the training, convert the jupyter notebook train.ipynb to a script train.py (jupyter nbconvert --to script train.ipynb) and copy the whole folder to the virtual machine (database included). Clone a new implementation of Mask_rcnn in the virtual machine (the one in the setup guide has been modified) and install it (python setup.py install in Mask_rcnn folder). Run the script on the virtual machine. Weights will be saved in thisfolder/logs/Config1/mask_rcnn_slices.h5 and tensorboard logs will be saved in a new folder in Config1. To visualize the training curves, download from the machine the tensorboard logs and launch tensorboard.

You will need the weights to run validation and whole image results. If you don't want to follow the process, we provide a link to the trained weights.

## validation

Download the weights from the machine/from the link into logs/Config1 and run the validation.ipynb notebook. For final whole image results, run the whole_image.ipynb notebook (our "run.ipynb")

## Links

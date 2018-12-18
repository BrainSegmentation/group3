# group3
group3 segmentation

## create dataset
the create_database.py script should patch the images contained in data. It will
take quite a lot of time. Moreover, you should manually move the result in two
directories, database/train and database/validate. You should respect the
structure created by create_database.py, i.e. patch j from image i should be in
 the path database/train/image_i/patch_j
## training
convert the jupyter notebook to a script and run it in the virtual machine.
## validation
download the weights in logs/Config1 and run the notebook. I've written a
modification of the visualize.py in maskrcnn to be able to visualize images
with multiple channels. Reinstall mrcnn with this modification.

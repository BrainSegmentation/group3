
# coding: utf-8

# In[ ]:


from mrcnn import utils
import numpy as np
import os
import skimage


# In[ ]:


class SlicesDataset(utils.Dataset):

    def load_slices(self, dataset_dir, n_images, n_patches, channels = ["base"]):
        """Load a subset of the Slices dataset.
        dataset_dir: Root directory of the dataset.
        n_images: number of images to load. Will load in os.listdir list order.
        n_partitions: number of equdistant partitions into which to divide the image.
        channels: list of strings indicating channels to be stacked in the image.
        currently "base", "mf", and "edges" can be arbitrarily stacked.
        """
        
        self.add_class("slices", 1, "tissue")
        self.add_class("slices", 2, "mag")
        
        image_list = os.listdir(dataset_dir)
        image_counter = 0
        patch_counter = 0
        for i in range(n_images):
            image_path = os.path.join(dataset_dir,image_list[i])
            patch_list = os.listdir(image_path)
            print(f"processing: image {i}")    
            for j in range(n_patches):
                patch_path = os.path.join(image_path, patch_list[j])
                patch_image_path = os.path.join(patch_path,"images")
                file_list = os.listdir(patch_image_path)
                image_file_path = os.path.join(patch_image_path,file_list[0])
                image = skimage.io.imread(image_file_path)
                height, width = image.shape
            
                self.add_image(
                    "slices",
                    image_id = patch_counter,
                    path = patch_path,
                    width = width, height = height,
                    channels = channels,
                )
                patch_counter +=1
            


    def load_image(self, image_id):
        """Returns a given image."""
        info = self.image_info[image_id]
        patch_path = info['path']
        width = info['width']
        height = info['height']
        impath = os.path.join(patch_path,"images")
        file_list = os.listdir(impath)         
        channels = info['channels']
        image = []
        for channel in channels:
            if channel == "none":
                channel_image = skimage.img_as_ubyte(np.zeros( (height,width) ) )
            else:
                channel_image_name = [x for x in file_list if channel in x][0] 
                channel_image_path = os.path.join(impath, channel_image_name)
                channel_image = skimage.io.imread(channel_image_path)
                channel_image = skimage.img_as_ubyte(channel_image)
            image.append(channel_image)
        image = np.stack(image, axis=2)
        return image
    
    def load_mask(self, image_id):
        """Generate instance masks for the given image ID.
        """
        info = self.image_info[image_id]
        patch_path = info['path']
        height = info['height']
        width = info['width']
        mag_path = os.path.join(patch_path,"mag")
        tissue_path = os.path.join(patch_path,"tissue")
        mag_mask_list = os.listdir(mag_path)
        tissue_mask_list = os.listdir(tissue_path)
        classes = []
        masks = []
        if mag_mask_list:
            for filename in mag_mask_list:
                a = os.path.join(mag_path,filename)
                masks.append(skimage.io.imread(a).astype(bool))
                classes.append(2)
        if tissue_mask_list:
            for filename in tissue_mask_list:
                a = os.path.join(tissue_path,filename)
                masks.append(skimage.io.imread(a).astype(bool))
                classes.append(1)
        return np.stack(masks,axis=2), np.asarray(classes).astype(int)


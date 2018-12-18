
# coding: utf-8

# In[58]:


import skimage
import os 
import sys
import numpy as np


# In[60]:


def import_txt(path):
    with open(path) as p:
            positions = []
            for line in p:
                coord = [str(x) for x in line.split()]
                for i in range(len(coord)):
                    txt = str(coord[i])
                    x, y = txt.split(',')
                    coord[i] = [int(int(x) / 3), int(int(y) / 3)]
                positions.append(coord)
            positions = np.asarray(positions)
    return positions
      


# In[61]:


def is_not_empty(positions_t, positions_m, boundaries):
    tflag=False
    mflag=False
    for l in range(len(positions_t)):
        barray1=( [boundaries[0][0] < positions_t[l,j,1] < boundaries[0][1] and
                 boundaries[1][0] < positions_t[l,j,0] < boundaries[1][1] 
                 for j in range(positions_t.shape[1]) ] )
        if any(barray1):
            tflag=True
            break
    for l in range(len(positions_m)):
        barray2=( [boundaries[0][0] < positions_m[l,j,1] < boundaries[0][1] and
                 boundaries[1][0] < positions_m[l,j,0] < boundaries[1][1] 
                 for j in range(positions_m.shape[1]) ] )
        if any(barray2):
            mflag=True
            break
    return tflag, mflag


# In[62]:


def create_database(data_in_path, data_out_path, patch_dimensions, patch_number, inference = False, ovl = 0, save_all = True ):
    """inpath is the path to the original database.
    outpath is where to save the cut images along with their masks.
    patch_dimensions is a len-2 list with y (first!) and x dimensions of each patch.
    pathc_number is the number of patches to be created.
    inference is a flag which tells if patches have to be created for inference mode. If it's false, image is cropped randomly."""
    #### Create path and check whether it contains a database. If so, ask for reset.
    ## (To be done)
    if not os.path.exists(data_out_path):
        os.makedirs(data_out_path)
    #### Training - random crop mode.    
    if inference == False:
        image_list = os.listdir(data_in_path) 
        image_counter = 0
        for folder in image_list:
            print("Processing ", folder)
            data_path = os.path.join(data_in_path, folder)
            file_list = os.listdir(data_path)
            image_files = [x for x in file_list if x.endswith('.tif') ]
            ic_image_file = [x for x in image_files if "intensityCorrected" in x ]
            mf_image_file = [x for x in image_files if "magFluo" in x ]
            mag_files = [x for x in file_list if "mag" in x and x.endswith('.txt')] 
            tissue_files = [x for x in file_list if "tissue" in x and x.endswith('.txt')]
            ic_image_path = os.path.join(data_path,ic_image_file[0])
            mf_image_path = os.path.join(data_path,mf_image_file[0])
            mag_path = os.path.join(data_path,mag_files[0])
            tissue_path = os.path.join(data_path,tissue_files[0])
            ic_image = skimage.io.imread(ic_image_path)
            mf_image = skimage.io.imread(mf_image_path)
            ic_image_edge = skimage.img_as_ubyte(skimage.filters.sobel(ic_image))
            height, width = ic_image.shape
            positions_t = import_txt(tissue_path)
            positions_m = import_txt(mag_path)
            patch_counter = 0
            image_path = os.path.join(data_out_path,f"image_{image_counter}")
            if not os.path.exists(image_path):
                    os.makedirs(image_path)
            while (patch_counter < patch_number):
                patch_path = os.path.join(image_path,f"patch_{patch_counter}")
                
                x1 = np.random.randint(0, height-patch_dimensions[0])
                x2 = np.random.randint(0, width-patch_dimensions[1])
                
                patch_base = ic_image[x1:x1+patch_dimensions[0],x2:x2+patch_dimensions[1]]
                patch_mf = mf_image[x1:x1+patch_dimensions[0],x2:x2+patch_dimensions[1]]
                patch_edges = ic_image_edge[x1:x1+patch_dimensions[0],x2:x2+patch_dimensions[1]]
                
                boundaries = [[x1,x1+patch_dimensions[0]], [x2,x2+patch_dimensions[1]]]
                tflag, mflag = is_not_empty(positions_t, positions_m, boundaries)
                
                if tflag or mflag:
                    if not os.path.exists(patch_path):
                        os.makedirs(patch_path)
                    patch_image_path = os.path.join(patch_path,"images")
                    if not os.path.exists(patch_image_path):
                        os.makedirs(patch_image_path)
                    patch_base_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_base.tif")
                    patch_mf_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_mf.tif")
                    patch_edges_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_edges.tif")
                    
                    skimage.io.imsave(patch_base_save_path, patch_base)
                    skimage.io.imsave(patch_mf_save_path, patch_mf)
                    skimage.io.imsave(patch_edges_save_path, patch_edges)
                    
                    t_mask_counter = 0
                    m_mask_counter = 0
                    
                    mag_mask_path = os.path.join(patch_path,"mag")
                    tissue_mask_path = os.path.join(patch_path,"tissue")
                    if not os.path.exists(mag_mask_path):
                        os.makedirs(mag_mask_path)
                    if not os.path.exists(tissue_mask_path):
                        os.makedirs(tissue_mask_path)
                    for i in range(len(positions_t)):
                        t_label_mask = np.zeros( (ic_image.shape[0], ic_image.shape[1]) ).astype(np.uint8)
                        t_pixels = skimage.draw.polygon(positions_t[i,:,1], positions_t[i,:,0])
                        skimage.draw.set_color(t_label_mask, t_pixels, 255)
                        t_mask_save_path = os.path.join(tissue_mask_path,f"tissue_mask_{t_mask_counter}.tif")
                        t_patch_mask = t_label_mask[ boundaries[0][0]:boundaries[0][1], boundaries[1][0]:boundaries[1][1] ]
                        if t_patch_mask.any():
                            skimage.io.imsave(t_mask_save_path,t_patch_mask)
                            t_mask_counter = t_mask_counter + 1
                    for i in range(len(positions_m)):
                        m_label_mask = np.zeros( (ic_image.shape[0], ic_image.shape[1]) ).astype(np.uint8)
                        m_pixels = skimage.draw.polygon(positions_m[i,:,1], positions_m[i,:,0])
                        skimage.draw.set_color(m_label_mask, m_pixels, 255)
                        m_mask_save_path = os.path.join(mag_mask_path,f"mag_mask_{m_mask_counter}.tif")
                        m_patch_mask = m_label_mask[ boundaries[0][0]:boundaries[0][1], boundaries[1][0]:boundaries[1][1] ]
                        if m_patch_mask.any():
                            skimage.io.imsave(m_mask_save_path,m_patch_mask)
                            m_mask_counter = m_mask_counter + 1
                    patch_counter = patch_counter + 1
            image_counter = image_counter + 1 
            
    if inference == True:
        image_list = os.listdir(data_in_path) 
        image_counter = 0
        for folder in image_list:
            print("Processing ", folder)
            data_path = os.path.join(data_in_path, folder)
            
            file_list = os.listdir(data_path)
            
            image_files = [x for x in file_list if x.endswith('.tif') ]
            ic_image_file = [x for x in image_files if "intensityCorrected" in x ]
            mf_image_file = [x for x in image_files if "magFluo" in x ]
            mag_files = [x for x in file_list if "mag" in x and x.endswith('.txt')] 
            tissue_files = [x for x in file_list if "tissue" in x and x.endswith('.txt')]
            
            ic_image_path = os.path.join(data_path,ic_image_file[0])
            mf_image_path = os.path.join(data_path,mf_image_file[0])
            mag_path = os.path.join(data_path,mag_files[0])
            tissue_path = os.path.join(data_path,tissue_files[0])
            
            ic_image = skimage.io.imread(ic_image_path)
            mf_image = skimage.io.imread(mf_image_path)
            ic_image_edge = skimage.img_as_ubyte(skimage.filters.sobel(ic_image))
            
            height, width = ic_image.shape
            
            positions_t = import_txt(tissue_path)
            positions_m = import_txt(mag_path)
            
            image_path = os.path.join(data_out_path,f"image_{image_counter}")
            if not os.path.exists(image_path):
                    os.makedirs(image_path)
            # patch_number = int(height/patch_dimensions[0])*int(width/patch_dimensions[1]) 

            patch_counter = 0
            counter_x1 = 0
            counter_x2 = 0
            limit_x1 = False
            limit_x2 = False

            while(limit_x1 == False):

                limit_x2 = False
                while(limit_x2 == False):


                    start_x2 = patch_dimensions[1]*counter_x2 - ovl*counter_x2
                    if start_x2 < 0:
                        start_x2 = 0

                    start_x1 = patch_dimensions[0]*counter_x1 - ovl*counter_x1
                    if start_x1 < 0:
                        start_x1 = 0

                    end_x2 = start_x2 + patch_dimensions[1]
                    if start_x2 + patch_dimensions[1] > width:
                        end_x2 = width
                        start_x2 = width - patch_dimensions[1]

                    end_x1 = start_x1 + patch_dimensions[0]
                    if start_x1 + patch_dimensions[0] > height:
                        end_x1 = height
                        start_x1 = height - patch_dimensions[0]

                    patch_base = ic_image[start_x1:end_x1,start_x2:end_x2]
                    patch_mf = mf_image[start_x1:end_x1,start_x2:end_x2]
                    patch_edges = ic_image_edge[start_x1:end_x1,start_x2:end_x2]
                    
                    boundaries = [[start_x1 , end_x1], [start_x2 , end_x2]]
                    tflag, mflag = is_not_empty(positions_t, positions_m, boundaries)

                    if tflag or mflag or save_all:
                        patch_path = os.path.join(image_path,f"patch_{counter_x1}_{counter_x2}")
                        if not os.path.exists(patch_path):
                            os.makedirs(patch_path)

                        patch_image_path = os.path.join(patch_path,"images")
                        if not os.path.exists(patch_image_path):
                            os.makedirs(patch_image_path)
                        
                        patch_base_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_base.tif")
                        patch_mf_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_mf.tif")
                        patch_edges_save_path = os.path.join(patch_image_path,f"patch_{patch_counter}_edges.tif")

                        skimage.io.imsave(patch_base_save_path, patch_base)
                        skimage.io.imsave(patch_mf_save_path, patch_mf)
                        skimage.io.imsave(patch_edges_save_path, patch_edges)

                        t_mask_counter = 0
                        m_mask_counter = 0

                        mag_mask_path = os.path.join(patch_path,"mag")
                        tissue_mask_path = os.path.join(patch_path,"tissue")
                        if not os.path.exists(mag_mask_path):
                            os.makedirs(mag_mask_path)
                        if not os.path.exists(tissue_mask_path):
                            os.makedirs(tissue_mask_path)
                        for i in range(len(positions_t)):
                            t_label_mask = np.zeros( (ic_image.shape[0], ic_image.shape[1]) ).astype(np.uint8)
                            t_pixels = skimage.draw.polygon(positions_t[i,:,1], positions_t[i,:,0])
                            skimage.draw.set_color(t_label_mask, t_pixels, 255)
                            t_mask_save_path = os.path.join(tissue_mask_path,f"tissue_mask_{t_mask_counter}.tif")
                            t_patch_mask = t_label_mask[ boundaries[0][0]:boundaries[0][1], boundaries[1][0]:boundaries[1][1] ]
                            if t_patch_mask.any():
                                skimage.io.imsave(t_mask_save_path,t_patch_mask)
                                t_mask_counter = t_mask_counter + 1
                        for i in range(len(positions_m)):
                            m_label_mask = np.zeros( (ic_image.shape[0], ic_image.shape[1]) ).astype(np.uint8)
                            m_pixels = skimage.draw.polygon(positions_m[i,:,1], positions_m[i,:,0])
                            skimage.draw.set_color(m_label_mask, m_pixels, 255)
                            m_mask_save_path = os.path.join(mag_mask_path,f"mag_mask_{m_mask_counter}.tif")
                            m_patch_mask = m_label_mask[ boundaries[0][0]:boundaries[0][1], boundaries[1][0]:boundaries[1][1] ]
                            if m_patch_mask.any():
                                skimage.io.imsave(m_mask_save_path,m_patch_mask)
                                m_mask_counter = m_mask_counter + 1
                    patch_counter = patch_counter + 1
                    counter_x2 = counter_x2 + 1

                    if end_x2 == width:
                        counter_x2 = 0
                        limit_x2 = True
                        counter_x1 = counter_x1 + 1

                if limit_x2 == True and end_x1 == height:
                    limit_x1 = True

            image_counter = image_counter + 1             


import numpy as np
import skimage
import os
import time
import sys

def calculate_centroid(image):
    """ this function calculates the centroid of a mask"""    
    
    # calculate x_2
    
    cum_sum = 0
    for i in range(image.shape[1]):
        cum_sum = cum_sum + np.sum(image[:,i].astype(int))*(i+1)
    centroid_2 = int(cum_sum / np.sum(image.astype(int))) - 1
    
    # calculate x_1
    cum_sum = 0
    for j in range(image.shape[0]):
        cum_sum = cum_sum + np.sum(image[j,:].astype(int))*(j+1)
    centroid_1 = int(cum_sum / np.sum(image.astype(int))) - 1
    
    centroid = np.asarray([centroid_1 , centroid_2])
    
    return centroid



def global_coord(centroid , coord_patch):
    """convert local patch coordinates to global coordinates"""
    
    g_centroid = np.asarray([0,0])
    g_centroid[0] = centroid[0] + coord_patch[0]
    g_centroid[1] = centroid[1] + coord_patch[1]
    
    return g_centroid

def has_overlap(mask_1,mask_2,pd):
    """Check if two patches have a possible overlap. pd is the patch dimensions"""
    return (np.absolute(mask_1['coord'] - mask_2['coord'])<pd).all()

def has_conflict(mask_1,mask_2,pd):
    """Check if two masks have an effective conflict. pd is the patch dimensions
    Conflict is present when more than 10 pixels coincide."""
    
    
    overlap_coordinates = [[0,0],[0,0]]
    overlap_in_image_1 = [[0,0],[0,0]]
    overlap_in_image_2 = [[0,0],[0,0]]
    
    # overlap global coordinates
    
    overlap_coordinates[0][0] = np.max([mask_1['coord'][0],mask_2['coord'][0]])
    overlap_coordinates[0][1] = np.min([mask_1['coord'][0] + pd[0], mask_2['coord'][0] + pd[0]])
    overlap_coordinates[1][0] = np.max([mask_1['coord'][1],mask_2['coord'][1]])
    overlap_coordinates[1][1] = np.min([mask_1['coord'][1] + pd[1], mask_2['coord'][1] + pd[1]])
    
    # overlap local coordinates
    
    overlap_in_image_1[0][0] = overlap_coordinates[0][0] - mask_1['coord'][0]
    overlap_in_image_1[0][1] = overlap_coordinates[0][1] - mask_1['coord'][0]
    overlap_in_image_1[1][0] = overlap_coordinates[1][0] - mask_1['coord'][1]
    overlap_in_image_1[1][1] = overlap_coordinates[1][1] - mask_1['coord'][1]
    
    overlap_in_image_2[0][0] = overlap_coordinates[0][0] - mask_2['coord'][0]
    overlap_in_image_2[0][1] = overlap_coordinates[0][1] - mask_2['coord'][0]
    overlap_in_image_2[1][0] = overlap_coordinates[1][0] - mask_2['coord'][1]
    overlap_in_image_2[1][1] = overlap_coordinates[1][1] - mask_2['coord'][1]

    # extract overlap.
    
    overlap1 = mask_1['mask'][overlap_in_image_1[0][0]:overlap_in_image_1[0][1],overlap_in_image_1[1][0]:overlap_in_image_1[1][1]]
    overlap2 = mask_2['mask'][overlap_in_image_2[0][0]:overlap_in_image_2[0][1],overlap_in_image_2[1][0]:overlap_in_image_2[1][1]]

    return np.sum(overlap1*overlap2)>10


def model_confl(model , image , patch_dimensions , min_ovl, suppress_over_mean = False, suppression_parameter = 3):
    """ This function applies a model over an image, dividing it in patches. Returns centroids of tissue and
    magnetic masks, and tissue part orientation. Orientation is inferred from nearest mag mask and returned as a
    versor.
    model contains the keras model classify the results with
    image contains the whole image to be segmented
    patch_dimensions contains the dimension of the single patch - [512,512] for our model
    min_ovl is the minimum overlap to use in the validation to be sure that no slice is lost. should be 
    the maximum possible slice dimension in pixels.
    suppress_over_mean is a flag that can be used to remove centroids which belong to mask which have a too
    small area. Used to remove corner misdetection.
    In this case masks are removed when they have area < mean_area/suppression_parameter."""
    
    # define result function
    
    function = lambda image: model.detect([image])[0]
    interpatch = [0,0]
    
    # calculate new overlap to patch the image to have equidistant patches
    # a few pixels are lost at the end of each image due to rounding errors.
    
    height = image.shape[0]
    width = image.shape[1]
    
    n_patches = np.asarray([0,0])
    
    n_patches[0] = int(np.ceil( (height - min_ovl)/(patch_dimensions[0] - min_ovl)))
    n_patches[1] = int(np.ceil( (width - min_ovl)/(patch_dimensions[1] - min_ovl)))
    
    interpatch[0] = int(np.floor( (height - patch_dimensions[0]) / (n_patches[0] - 1) ) )
    interpatch[1] = int(np.floor( (width - patch_dimensions[0]) / (n_patches[1] - 1) ) )
    
    patch_counter = 0
    
    ############################### classification
    
    print('Applying model to image..')
    start = time.time()
    
    r = []
    for i in range(n_patches[0]):            
        for j in range(n_patches[1]):
            
            # define patch boundaries and coordinates
            
            start_x1 = interpatch[0] * i
            start_x2 = interpatch[1] * j
            
            end_x1 = start_x1 + patch_dimensions[0]
            end_x2 = start_x2 + patch_dimensions[1]

            patch_image = image[start_x1 : end_x1 , start_x2 : end_x2, :]
            
            coord_patch = np.array([start_x1 , start_x2])
            
            # apply the model
            
            result = function(patch_image)
            
            # add patch coordinates to the result for our purposes
            
            result['coord'] = coord_patch
            
            r.append(result)
    
    # masks are now put singularly as new arrays into a separate list, and divided over their class
    
    masks = []
    
    for i in range(len(r)):
        for j in range(len(r[i]['class_ids'])):
            tmp = {}
            tmp['coord'] = r[i]['coord']
            tmp['class_id'] = r[i]['class_ids'][j]
            tmp['mask'] = r[i]['masks'][:,:,j]
            masks.append(tmp)
    
    tissue_masks = [mask for mask in masks if mask['class_id'] == 1]
    mag_masks = [mask for mask in masks if mask['class_id'] == 2]
    
    end = time.time()
    print(f'Done! Elapsed time : {end-start}')
    
    ################################### conflicts resolution
    
    print('Solving conflicts...')
    start = time.time()
    
    # every possible mask pair is checked within same class.
    
    for i in range(len(tissue_masks)):
        for j in range(i+1,len(tissue_masks)):
            if has_overlap(tissue_masks[i],tissue_masks[j],patch_dimensions):
                if has_conflict(tissue_masks[i],tissue_masks[j],patch_dimensions):
                    
                    # masks are removed according to which has biggest area.
                    # they are set to all zero images in order not to interfere with other conflicts.
                    
                    if np.sum(tissue_masks[i]['mask'])>np.sum(tissue_masks[j]['mask']):
                        tissue_masks[j]['mask'] = np.zeros(patch_dimensions).astype(bool)
                    else:
                        tissue_masks[i]['mask'] = np.zeros(patch_dimensions).astype(bool)
                        continue
    
    # remove empty masks
    
    tissue_masks = [mask for mask in tissue_masks if mask['mask'].any()]

    for i in range(len(mag_masks)):
        for j in range(i+1,len(mag_masks)):
            if has_overlap(mag_masks[i],mag_masks[j],patch_dimensions):
                if has_conflict(mag_masks[i],mag_masks[j],patch_dimensions):
                    
                    # masks are removed according to which has biggest area.
                    # they are set to all zero images in order not to interfere with other conflicts.
                    
                    if np.sum(mag_masks[i]['mask'])>np.sum(mag_masks[j]['mask']):
                        mag_masks[j]['mask'] = np.zeros(patch_dimensions).astype(bool)
                    else:
                        mag_masks[i]['mask'] = np.zeros(patch_dimensions).astype(bool)
                        continue
    
    # remove empty masks
    
    mag_masks = [mask for mask in mag_masks if mask['mask'].any()]
    
    end = time.time()
    print(f"Done! Elapsed time : {end-start}")
    
    ####################### centroids calculation 
    
    print('Calculating centroids and orientations...')
    start = time.time()
    
    
    centroids_tissue = []
    centroids_mag = []
    orientations = []
    tissue_area = []
    mag_area = []
    
    for i in range(len(tissue_masks)):
        
        # add centroid to mask
        
        tissue_masks[i]['centroid'] = []
        rel_centroid = calculate_centroid(tissue_masks[i]['mask'])
        tissue_masks[i]['centroid']=(global_coord(rel_centroid, tissue_masks[i]['coord']))
        
        # add centroid to centroid list
        
        centroids_tissue.append(tissue_masks[i]['centroid'])
        
        # add area to area list
        
        tissue_area.append(np.sum(tissue_masks[i]['mask']))
    
    #calculate mean area
    
    tissue_mean_area = np.mean(tissue_area) 
    
    if suppress_over_mean:
        
        # discard small masks
        
        centroids_tissue = [centroids_tissue[i] for i in range(len(centroids_tissue)) if tissue_area[i]>=tissue_mean_area/suppression_parameter]
    
    centroids_tissue = np.stack(centroids_tissue, axis=0)
    
    for i in range(len(mag_masks)):
        
        # add centroid to mask
        
        mag_masks[i]['centroid'] = []
        rel_centroid = calculate_centroid(mag_masks[i]['mask'])
        mag_masks[i]['centroid']=(global_coord(rel_centroid, mag_masks[i]['coord']))
        
        # add centroid to centroid list
        
        centroids_mag.append(mag_masks[i]['centroid'])
        
        # add area to area list
        
        mag_area.append(np.sum(mag_masks[i]['mask']))
    
    # calculate mean area
    
    mag_mean_area = np.mean(mag_area) 
    
    if suppress_over_mean:
        
        # discard small masks
        
        centroids_mag = [centroids_mag[i] for i in range(len(centroids_mag)) if mag_area[i]>=mag_mean_area/suppression_parameter]
    
    centroids_mag = np.stack(centroids_mag, axis=0)    
    
    ###################### orientation calculation 
    
    
    for i in range(len(centroids_tissue)):
        
        # find closest mag centroid and infer orientation from it.
        
        min_distance = float('inf')
        
        for j in range(len(centroids_mag)):
            
            distance = np.linalg.norm(centroids_tissue[i] - centroids_mag[j])
            
            if distance < min_distance:
                index = j
                min_distance = distance
        
        vector = (centroids_tissue[i]-centroids_mag[index])
        
        orientations.append(vector/np.linalg.norm(vector))
    
    orientations = np.stack(orientations)
    
    end = time.time()
    print(f"Done! Elapsed time : {end-start}")
    
    return centroids_tissue, centroids_mag, orientations
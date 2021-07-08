# import required libraries for image handling
import numpy as np
import cv2

# INITIALIZE CONSTANTS
scale_subregion = float(3) / 4                 # scale to crop new region
scale_mask = float(1) / (scale_subregion * 4)    # scale to get offset

# define function to get IOU value
# INTERSECTION OVER UNION (IOU)
def calculate_IOU(mask_region, mask_target_object):
    """
        input : numpy array of region mask, numpy array of target object mask
        output : IOU value between (mask_region and target object mask)
    """
    mask_target_object *= 1.0    # make it float
    
    # do AND and OR operation
    image_and = cv2.bitwise_and(mask_region, mask_target_object)
    image_or = cv2.bitwise_or(mask_region, mask_target_object)
    
    # count total cells with 1 (non zero)
    i = np.count_nonzero(image_and)    # any one nonzero
    j = np.count_nonzero(image_or)     # both nonzero
    
    # calculate IOU value
    IOU = float(float(i)/float(j))
    
    # return IOU value 
    return IOU
    

# define function to find the max IOU object mask
def max_BB(masks_objects, mask_region, classes_objects, class_object):
    """
        input : numpy array of objects masks, numpy array of region mask, list of classes of objects present in image, target object class
        output: return max IOU value for any object mask
    """
    # get total objects count
    objects_count = len(classes_objects)
    
    # initialize max IOU
    max_IOU = 0.0
    
    # for each object mask
    for k in range(objects_count):
        # check if not target object mask
        if classes_objects[k] != class_object:
            continue
        
        # otherwise find IOU
        mask_target_object = masks_objects[:, :, k]   # target object mask
        IOU = calculate_IOU(mask_region, mask_target_object)
        
        # check if current has max IOU
        if max_IOU < IOU:
            max_IOU = IOU
        
    # return max IOU value
    return max_IOU
        
# define function to crop current image 
def crop_current_image(original_shape, offset, current_image, current_shape, action):
    """
        input : original shape (of whole image), offset to original image, current region image, current region shape, action to perform
        output : offset of region cropped, new region image, new region shape, and new region mask
    """
    # Initialize mask for whole image
    mask_region = np.zeros(original_shape)
    
    # calculate new region shape   (h * s, w * s)
    new_shape = (current_shape[0] * scale_subregion, current_shape[1] * scale_subregion )
    
    # now find the offset and auxiallary offset
    if action == 1:
        offset_aux = (0 ,0)
    elif action == 2:
        offset_aux = (0, new_shape[1] * scale_mask)
        offset = (offset[0], offset[1] + new_shape[1] * scale_mask)
    elif action == 3:
        offset_aux = (new_shape[0] * scale_mask, 0)
        offset = (offset[0] + new_shape[0] * scale_mask, offset[1])
    elif action == 4:
        offset_aux = (new_shape[0] * scale_mask, new_shape[1] * scale_mask)
        offset = (offset[0] + new_shape[0] * scale_mask, offset[1] + new_shape[1] * scale_mask)
    elif action == 5:
        offset_aux = (new_shape[0] * scale_mask / 2, new_shape[0] * scale_mask / 2)
        offset = (offset[0] + new_shape[0] * scale_mask / 2, offset[1] + new_shape[0] * scale_mask / 2)
        
    # find new region image from current image
    new_reg_img = current_image[int(offset_aux[0]):int(offset_aux[0] + new_shape[0]), int(offset_aux[1]):int(offset_aux[1] + new_shape[1])]   # copy specific region
    
    # find new region image mask 
    mask_region[int(offset[0]):int(offset[0] + new_shape[0]), int(offset[1]):int(offset[1] + new_shape[1])] = 1
    
    # return new region variables
    return offset, new_reg_img, new_shape, mask_region
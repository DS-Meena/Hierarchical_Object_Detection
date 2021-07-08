# import the required libaries for image handling
import numpy as np
from PIL import ImageDraw
from PIL import Image
import cv2

# define function to load image names
def load_images_names_in_data_set(data_set_name, path_voc):
    """
        input : object dataset name and dataset path
        output : list of all image names in object dataset
    """
    # define the file path to given object dataset
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    
    # open the file and read all images 
    f = open(file_path)
    image = f.readlines()
    
    # extract all image names
    image_names = [x.strip('\n') for x in image]
    image_names = [x.split(None, 1)[0] for x in image_names]
    
    # return image names list
    return image_names

# define function to load image labels
def load_images_labels_in_data_set(data_set_name, path_voc):
    """
        input : object dataset name and dataset path
        output : list of all image labels in object dataset
    """
    # define the file path to given object dataset
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    
    # open the file and read all images
    f = open(file_path)
    images = f.readlines()
    
    # extract all images labels
    image_labels = [x.split(None, 1)[1] for x in images]
    image_labels = [x.strip('\n').strip(None) for x in image_labels]
    
    # return list of image labels
    return image_labels

# define function to load all images
def get_all_images(image_names, path_voc):
    """
        input : list of names of required images and path to dataset
        output : list of images
    """
    images = []
    
    # get corresponding images of image names 
    for image_name in image_names:
        # define path to image
        image_path = path_voc + '/JPEGImages/' + image_name + '.jpg'
        
        # read the image
        images.append(cv2.imread(image_path))   # read using cv2
#         images.append(Image.open(image_path))
        
    # return list of images
    return images
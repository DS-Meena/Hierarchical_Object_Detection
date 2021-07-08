# import required libraries for annotations handling
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# define function to get bb of image
def get_bbs_coordinates(image_name, voc_path):
    """
        input : image name and dataset path
        output : category and bb of image
    """
    # define path to image annotations
    xml_path = voc_path + '/Annotations/' + image_name + '.xml'
    
    # pares the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # initialize list to store coordinates
    names = []     # names of object present in image
    x_min = []     # coordinates of object
    x_max = []
    y_min = []
    y_max = []
    
    # get the xmin, xmax, ymin, ymax, names for image
    for child in root:
        if child.tag == 'object':            # if tag of object
            for child2 in child:
                if child2.tag == 'name':     # if tag of object name
                    names.append(child2.text)
                elif child2.tag == 'bndbox':      # if tag of bounding box
                    for child3 in child2:
                        if child3.tag == 'xmin':     # get the bb coordinates
                            x_min.append(child3.text)
                        elif child3.tag == 'xmax':
                            x_max.append(child3.text)
                        elif child3.tag == 'ymin':
                            y_min.append(child3.text)
                        elif child3.tag == 'ymax':
                            y_max.append(child3.text)
    
    # create numpy array of size = no of objects
    category_and_bb = np.zeros([np.size(names), 5])
    
    for i in range(np.size(names)):
        # get id of class name = names[i]
        category_and_bb[i][0] = get_id_of_class_name(names[i])
        category_and_bb[i][1] = x_min[i]
        category_and_bb[i][2] = x_max[i]
        category_and_bb[i][3] = y_min[i]
        category_and_bb[i][4] = y_max[i]
    
    # return category and bounding box
    return category_and_bb

# function to get the masks for the objects 
def get_bb_objects(annotation, image_shape):
    """
        input : annotation (names, coordinates) of an image and image shape
        output : masks (bb) for the objects in image (b\w)
    """
    # get no of objects
    objects_count = annotation.shape[0]
    
    # initialize a numpy array of zeros
    masks = np.zeros([image_shape[0], image_shape[1], objects_count])  # size = (H, W, objects_count)
    
    # for each object
    for i in range(objects_count):
        masks[int(annotation[i, 3]):int(annotation[i, 4]), int(annotation[i, 1]):int(annotation[i, 2]), i] = 1     # mark the object mask as 1
        
    # return the masks for current image
    return masks


# function to get id of the given class name
def get_id_of_class_name(class_name):
    """
        input : class name of image
        output : id of the given class name
    """
    # check all conditions 
    if class_name == 'aeroplane':
        return 1
    elif class_name == 'bicycle':
        return 2
    elif class_name == 'bird':
        return 3
    elif class_name == 'boat':
        return 4
    elif class_name == 'bottle':
        return 5
    elif class_name == 'bus':
        return 6
    elif class_name == 'car':
        return 7
    elif class_name == 'cat':
        return 8
    elif class_name == 'chair':
        return 9
    elif class_name == 'cow':
        return 10
    elif class_name == 'diningtable':
        return 11
    elif class_name == 'dog':
        return 12
    elif class_name == 'horse':
        return 13
    elif class_name == 'motorbike':
        return 14
    elif class_name == 'person':
        return 15
    elif class_name == 'pottedplant':
        return 16
    elif class_name == 'sheep':
        return 17
    elif class_name == 'sofa':
        return 18
    elif class_name == 'train':
        return 19
    elif class_name == 'tvmonitor':
        return 20
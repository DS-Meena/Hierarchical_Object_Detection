# import the required libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import numpy as np
import matplotlib.pyplot as plt


# define transform for the image
transform = transforms.Compose([
            transforms.ToPILImage(),         # convert to pil image
            transforms.Resize((224, 224)),    # resize image
            transforms.ToTensor(),           # convert numpy array to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # normalize
])

# function to load vgg16 model
def getVGG_16bn(path_vgg):
    """
        input : path to store the model
        output : vgg16 model
    """
    # if vgg16 avialable at path_vgg load it
    # otherwise download it from url & save at path_vgg
    state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', path_vgg)
    model = torchvision.models.vgg16_bn()
    model.load_state_dict(state_dict)
    
    # removing the classifier
    model_2 = list(model.children())[0]
    
    # return vgg16 model
    return model_2

# function to get the descriptor image for current image
def get_descriptor_image(current_image, model_vgg, dtype=torch.cuda.FloatTensor):
    """
        input : numpy array of current image, pretrained vgg16 model
        output : tensor of descriptor of image
    """
    # apply transform on the image
    img = transform(current_image)
    img = img.view(1, *img.shape)    # convert the dimesion
    
    # get the features from the image using vgg16
    feature = model_vgg(Variable(img).type(dtype))
    
    # return the descriptor features
    return feature.data
    
    

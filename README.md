# Hierarchical Object detection

 This is a implementation of research paper [Hierachical Object detection using Deep RL]. We have done it in Pytorch.  

## Setup Environment

Python 3.8.5 is used.
torch 1.8 with torchvision 0.9 versions are used.

Download the Pytorch setup (cuda supported version):

``conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch``

Other package requirements are in requirements.txt file.
You can install them using

``pip install -r requirements.txt`` command.

If it shows "Module Not found error" on any cell, you can simply use PIP command to install that package, like -> ``pip install package_name``

Note-
Some errors were encountered during import of libraries. As follows:

- to resolve the cv2 error, I used ``apt-get install -y python3-opencv``
- to resolve the ipywidgets issue, I used ``conda install -c conda-forge ipywidgets`` and ``jupyter nbextension enable --py widgetsnbextension``

## Download Dataset

we have only used Pascal VOC 2007 dataset during training and testing. But you can also use Pascal VOC 2012 dataset, both datasets have similar files and data structure.

- Set Kaggle Credentials

``kaggle_data={"username":"","key":""}
os.environ['KAGGLE_USERNAME']=kaggle_data["username"]
os.environ['KAGGLE_KEY']=kaggle_data["key"]``

- Download Pascal Voc Datasets

``! kaggle datasets download -d zaraks/pascal-voc-2007``

  and [optional]

``! kaggle datasets download -d huanghanchina/pascal-voc-2012``

- Unzip Datasets
``!unzip \*.zip``

Note - You can also follow above steps in train.ipynb, in Dataset Cells by uncommenting code.

## Setup DataFolder

Create Data and Models Folder. 
Create 2 sub Folders Train and Test inside Data Folder.

After unzipping Pascal voc 2007 zip files.

1. Open the voctrainval_06-nov-2007--> VOCdevkit --> Move the VOC2007 folder (containing Annotations, JpegImages folder, etc) to train folder created above.

2. Open the voctest_06-nov-2007 --> VOCdevkit --> Move the VOC2007 folder (containing Annotations, JpegImages folder, etc) to the test folder created above.

[optional]
Similarly, After unzipping Pascal voc 2012 folder.

1. Open the VOC2012 folder --> Move VOC2012 folder (containing Annotations, JpegImages folder, etc) to the train folder created above.

Note - Some folder names are reduntant, but before moving folder make sure it contains the annotations, JpegImages folder.

### For Custom Images and Walking Person video

1. Create Custom_Data Folder Inside test Folder.
2. Create 2 sub-folders inside Custom_Data folder, a)Images b) video
3. Inside Video Folder, create another sub-folder frames.
4. You can download the images for the Test/Custom_Data/Images sub-folder
from the repository.
5. Similarly, you can download the video of walking person for the Test/Custom_Data/Video sub-folder from the repository.


Final File structure:

./

    Data/
        Test/
            VOC2007/
                Annotations/
                SegmentationClass/
                JPEGImages/
                SegmentationObject/
                ImageSets/
                    Segmentation/
                    Main/
                    Layout/
            Custom_Data/
                Images/
                Video/
                    frames/
        Train/
            VOC2012/
                Annotations/
                SegmentationClass/
                JPEGImages/
                SegmentationObject/
                ImageSets/
                    Segmentation/
                    Action/
                    Main/
                    Layout/
            VOC2007/
                Annotations/
                SegmentationClass/
                JPEGImages/
                SegmentationObject/
                ImageSets/
                    Segmentation/
                    Main/
                    Layout/
    Models/

After This setup, you can try to run the Notebooks. and hopefully they should run without not too much library errors.

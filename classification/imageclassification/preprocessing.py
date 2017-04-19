import Image
from distlib import resources
from pkgutil import get_data

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from pkg_resources import *
import shutil


#Class to do some pre-processing on data
source = "resources"
destination = "training"

#apppath = os.path.dirname(os.path.abspath(__file__))
#appresources = resources.Resources(os.path.join(apppath, source))

#/home/prime/ann/untrodden/resources
# sampleImage = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"resources","training","s1","1.pgm"))
# img = load_img(sampleImage)
#
# image = Image.open(img,'rb')
# Image.Image.save(os.path.join(os.path.dirname(__file__),'..',"resources","training",img),'png')


source = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"test"))
training = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"resources","training"))
validation = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"resources","validation"))
test = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"resources","test"))

for folder in os.listdir(source):

    if str(folder).startswith("s"):
        folderName = os.path.join(source,folder)
        print "*" * 10
        print folder
        print folderName
        for image in os.listdir(folderName):
            #print image
            modifiedImage = Image.open(os.path.join(folderName,image)).convert('L')
            modifiedImage.save(os.path.join(folderName,image.split(".")[0]+'.png'),'png')
            #print modifiedImage

            if int(image.split(".")[0]) in xrange(1,7):
                #print image[0].split(".")[0]
                directory = os.path.join(training, folder)
                #print directory
                if not os.path.exists(directory):
                    os.makedirs(directory)
                shutil.move(os.path.join(folderName,image),directory)
            elif int(image.split(".")[0]) in xrange(7,9):
                directory = os.path.join(validation, folder)
                #print directory
                if not os.path.exists(directory):
                    os.makedirs(directory)
                shutil.move(os.path.join(folderName,image), directory)
            else:
                directory = os.path.join(test, folder)
                #print directory
                if not os.path.exists(directory):
                    os.makedirs(directory)
                shutil.move(os.path.join(folderName,image), directory)
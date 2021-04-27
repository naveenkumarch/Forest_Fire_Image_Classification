#!/usr/bin/env python
# coding: utf-8
"""
The script is used for generating randomly augmented data.  
keras experimental preprocessing layers are utilized in the program for randomly augmenting the data and saving it.

"""


import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import RandomContrast,RandomRotation,RandomCrop,RandomFlip

# Generating the sequential model with preprocessing layers
transform = tf.keras.Sequential([
    RandomRotation(0.2, fill_mode='reflect', interpolation='bilinear'),
    RandomFlip("horizontal_and_vertical"),
    RandomContrast(0.2),
    RandomCrop(220, 220)
])


# paths to read images and path to save images
folder_path = 'D:\MSC\Computer_Vision\Labs\Cropped_images'
Desired_no_of_files = 500

Save_folder = 'D:\MSC\Computer_Vision\Labs\Cropped_images'
# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


generated_files = 1
while generated_files <= Desired_no_of_files:
    # randomly selecting an image
    image_path = random.choice(images)

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Expanding the dimensions of the image for passing it on to keras preprocessing layers
    image = tf.expand_dims(image, 0)

    # Augmenting the image
    augment = transform(image)
    transformed_image = augment[0]

    new_file_path = '%s/augmented_image_%s.jpg' % (Save_folder, generated_files)
    tf.keras.preprocessing.image.save_img(new_file_path,np.array(transformed_image),file_format='png')

    generated_files += 1






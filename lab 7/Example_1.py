# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:27:26 2019

@author: hr17576
"""

import numpy as np
from vgg16 import VGG16
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions


model = VGG16(include_top=True, weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

model.summary()
model.layers[-1].get_config() 

#%% 

# This will remove the consideration of top layer
model = VGG16(weights='imagenet', include_top=False)
# After the above line there will be no flatten layer becuause top = False

model.summary()
model.layers[-1].get_config()

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)


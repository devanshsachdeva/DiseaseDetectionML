# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:49:04 2018

@author: shubham
"""
import random

from keras.models import Sequential
# above module is used for initializing the neural networks

from keras.layers import Convolution2D
# in above 2-D indicates that we are dealing with images

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# above module is used for adding ANN layer in the model

from keras.models import model_from_json

random.seed(1)

# Initialising the CNN
classifier = Sequential()
# above command indicates our model is a sequence of layers.

# Step 1 - Convolution layer
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# in above classifier.add means we are adding a layer to the model
# in above command 32 indicates no. of feature detectors(or filters) that we are going to apply 
# and (3,3) indicates no. of rows and columns in each feature detectors. 
# input_shape tells the shape of input image on which we are going to apply convolution operation.
# Here (64,64,3) indicates that each image will of size 64X64 and  3 indicates we are using coloured images. 
# activation = 'relu' means that we are applying rectified linear unit(relu) as the activation function.

# Step 2 - Pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# in above command pool_size indicates size of pooled matrix. By doing this we are reducing the size by 2. 

# Adding second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# above command is used for flattening the data

# Step 4 - Full connection. Adding ANN in the model
classifier.add(Dense(units = 128, activation = 'relu'))
# in above command units means no. of nodes in the hidden layer. It is a good practice to make no. of
# nodes in the hidden layer greater than 100. activaton function used is rectified linear unit(relu).

# Now finnally add output layer.
classifier.add(Dense(units = 2, activation = 'softmax'))
# in above command units = 2 is there because we have 2 classes and since we want probabilities of both
# classes so we have used softmax function as the activation function.

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# in above command we are basically applying stochastic gradient descent on the whole ANN model.
# optimiser = 'adam' indicates the algorithm that we want to use to find optimal set of weights in the
# neural networks. There are several types of stochastic gradient descent and the most efficient one
# is called adam.
# loss = 'category_crossentopy' corresponds to the loss function within stochastic gradient descent algo.
# since outcome variable has 2 classes so we have used categorical_crossentropy.
# metrics = ['accuracy'] indicates how you want to evaluate your model. We choose accuracy to evaluate
# our model.

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# above module is used for image augmentation.

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# in above basically we are saying that we will be applying above pre-processing on the images.
# pre-processing include scaling the pixel values between 0 and 255.
# applying zoom on the images.
# applying horizontal flip on the images. 

test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'categorical',
                                                 shuffle = False)
# the pre-processing that we stated above is now applied on the training dataset images.
# in bove target_size = (64,64) indicates that all images will be of size 64 X 64.
# by stating class_mode = 'categorical', it will automatically determine how many classes are there 
# depending on number of folders inside the 'training_set' folder.
# by stating shuffle = False, we are not shuffling the images.
 
testing_dataset = test_datagen.flow_from_directory('testing_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle = False)


# to know which label represents which eye disease, use below code
training_dataset.class_indices
# above will tell you the classes in the training_dataset

# for knowing classes in test dataset use code below
testing_dataset.class_indices

random.seed(0)

# now we will simply fit the model on the training dataset.
classifier.fit_generator(training_dataset,
                         steps_per_epoch = 14,
                         epochs = 25)
# in above we have selected 25 epochs i.e. 25 iterations. And in each iteration 14 images will be trained
# here we have taken steps_per_epoch = 14 because 14 images are there in training dataset


random.seed(2)

# to open the saved model use below comand
json_file = open('cnnmodel.json', 'r')
# cnnmodel.json file was there in the same directory, we have opened it.
loaded_model_json = json_file.read()
# in above command we have read the json_file.
json_file.close()
classifier = model_from_json(loaded_model_json)
# the model inside the json file is extracted.
classifier.load_weights("cnnmodel.h5")
# in above command we have loaded the weights on which model was built.
prob = classifier.predict_generator(testing_dataset)
n =  training_dataset.n
# n contains number of images in training set 
m = testing_dataset.n
# m contains number of images in testing set
for i in range(0,m):
    if(i % 2 != 0):
        if(prob[i,0] > prob[i,1]):
            prob[i,0] = prob[i,0] - 0.1
            prob[i,1] = prob[i,1] + 0.1
        else:
            prob[i,0] = prob[i,0] + 0.1
            prob[i,1] = prob[i,1] - 0.1
    else:
        if(prob[i,0] > prob[i,1]):
            prob[i,0] = prob[i,0] - 0.13
            prob[i,1] = prob[i,1] + 0.13
        else:
            prob[i,0] = prob[i,0] + 0.13
            prob[i,1] = prob[i,1] - 0.13
 
# Keras provides the ability to describe any model using JSON format with a to_json() function. 
# This can be saved to file and later loaded via the model_from_json() function that will create a 
# new model from the JSON specification.    
# if you want to save your model follow below lines
classifier_json = classifier.to_json()
# above command converts the model layers to json format.
with open("cnnmodel.json", "w") as json_file:
    json_file.write(classifier_json)
# in above command we opened an empty file named as cnnmodel.json in write mode and wrote the information 
# about model layers in it. 
    
classifier.save_weights("cnnmodel.h5") 
# in above command we saved the model weights. 
# Model weights are saved to HDF5 format. This is a grid format that is ideal for storing multi-dimensional 
# arrays of numbers.    
        

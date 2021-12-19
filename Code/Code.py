'''###   Assignment :- Face Recognition   ###'''

##  Using CNN

'''### 1. Image Processing  ###'''

## Specifying the folder where images are present
Dataset_Path='Code/Dataset'

from keras.preprocessing.image import ImageDataGenerator

## Process traning image 
training_data = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

## No change to testing image
testing_data = ImageDataGenerator()

print('')

## Generating the Training Data
training_set = training_data.flow_from_directory(
        Dataset_Path,
        target_size=(64, 64),
        batch_size=10,
        class_mode='categorical')

print('')

# Generating the Testing Data
testing_set = testing_data.flow_from_directory(
        Dataset_Path,
        target_size=(64, 64),
        batch_size=10,
        class_mode='categorical')

# Printing class labels for each face
testing_set.class_indices




'''###   Creat lookup table for all faces   ###'''

# class_indices have the numeric tag for each face
TrainClasses=training_set.class_indices

# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("\nMapping of Face and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\nThe Number of output neurons: ', OutputNeurons,'\n')




'''###    Create CNN model    ###'''

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import time

##  Initializing the Convolutional Neural Network
classifier= Sequential()

##  STEP--1 Convolution 1st
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))

##  STEP--2 MAX Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

##  ADDITIONAL LAYER of CONVOLUTION & MAX Pooling for better accuracy
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))

##  STEP--3 FLattening
classifier.add(Flatten())

##  STEP--4 Fully Connected Neural Network
classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(OutputNeurons, activation='softmax'))

##  Compiling the CNN
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

# Measuring the time taken by the model to train
StartTime=time.time()

# Starting the model training
classifier.fit(
                    training_set,
                    steps_per_epoch=6,                          # total_images/batch_of_training=60/10
                    epochs=10,
                    validation_data=testing_set,
                    validation_steps=6)

EndTime=time.time()
print("\n###   Total Time Taken for traning : ", round((EndTime-StartTime)/60), 'Minutes    ###\n')





'''###      Making predictions & Testing Module    ###'''

import numpy as np
from keras.preprocessing import image

ImagePath='Code/Dataset/Shah Rukh Khan/Khan_7.jpg'
test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image,verbose=0)

print('\nImage belong to :- ',ResultMap[np.argmax(result)],'\n\n')

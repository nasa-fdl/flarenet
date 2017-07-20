"""
This is a brain-dead auto-encoder for the 128X128 magnetograms as provided by Mark.
You will need to update the data_directory to point at the data.

Adapted from https://blog.keras.io/building-autoencoders-in-keras.html
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
import os
import random

data_directory = "/home/smcgregor/projects/solar-forecast/datasets/sdo_128/bin/"
seed = 0
random.seed(seed)

input_img = Input(shape=(128, 128, 8))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (128, 128, 16)
x = MaxPooling2D((2, 2), padding='same')(x)                          # (64, 64, 16)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (64, 64, 8)
x = MaxPooling2D((2, 2), padding='same')(x)                          # (32, 32, 8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (32, 32, 8)
encoded = MaxPooling2D((2, 2), padding='same')(x)                    # (16, 16, 8)


x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)    # (16, 16, 8)
x = UpSampling2D((2, 2))(x)                                          # (32, 32, 8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (32, 32, 8)
x = UpSampling2D((2, 2))(x)                                          # (64, 64, 8)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)                         # (64, 64, 16)
x = UpSampling2D((2, 2))(x)                                          # (128, 128, 16)
decoded = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x) # changed to 8

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Uncomment to plot the network architecture
#from keras.utils import plot_model
#plot_model(autoencoder, to_file='model2.png', show_shapes=True)
#exit()

# get a directory listing of the sdo mnist data
filenames = os.listdir(data_directory)

# remove a random subset of the file name list and make them the test set
random.shuffle(filenames)
train_files = filenames[:-100]
test_files = filenames[-100:]

# pack the x_train from the train set
x_train = []
for f in train_files:
    print "adding to training set: " + f
    data = np.memmap(data_directory + f, dtype='uint8', mode='r', shape=(128,128,8))
    x_train.append(data[:].copy())
x_train = np.asarray(x_train)
    
x_test = []
for f in test_files:
    print "adding to testing set: " + f
    data = np.memmap(data_directory + f, dtype='uint8', mode='r', shape=(128,128,8))
    x_test.append(data[:].copy())
x_test = np.asarray(x_test)
    
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 128, 128, 8))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 128, 128, 8))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=5000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

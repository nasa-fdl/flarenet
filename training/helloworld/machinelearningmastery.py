"""
This is a simplistic model to work through so you can know the GPUs, etc, are working for you.
Since the dataset is small, it is already downloaded for you.

Adapted from http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)  # fix random seed for reproducibility

data_source_path = "../../datasets/helloworld/pima-indians-diabetes.csv"
data_file = open(data_source_path, "rb")

# load pima indians dataset
dataset = numpy.loadtxt(data_file, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#with tf.device('/gpu:0'):

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=1000, batch_size=40)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

data_file.close()

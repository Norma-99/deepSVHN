#Load SVHN dataset from Tensorflow
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as pyplot
import cv2
from datetime import datetime

LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 1e-4
IMAGE_SIZE = 255.0
PATH = './logs/networks/{}'.format(datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
DATA_SIZE = 100

def scheduler(epoch):
    lr = float(LEARNING_RATE * tf.math.exp(-(epoch-1)*LEARNING_RATE_DECAY))
    print(lr)
    return lr

print('loading dataset...')

#Load training data
print('loading training data...')
train_ds = tfds.load(name="svhn_cropped", split=tfds.Split.TRAIN)
train_list = list(tfds.as_numpy(train_ds))

#Load test data
print('loading test data...')
test_ds = tfds.load(name="svhn_cropped", split=tfds.Split.TEST)
test_list = list(tfds.as_numpy(test_ds))

#x_train is the data for training the dataset
#y_train is the set of labels to all the data in x_train
x_train = list()
y_train = list()

#x_test is de data for testing the dataset
#y_test is the set of labels to all the data in x_test
x_test = list()
y_test = list()

#Creates the data to a generator of numpy arrays
for pair in train_list [:70000]:
    #Add to list
    x_train.append(pair['image'])
    y_train.append(pair['label'])

for pair in test_list [:10000]:
    #Add to list
    x_test.append(pair['image'])
    y_test.append(pair['label'])

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

#normalize data
x_train, x_test = x_train / IMAGE_SIZE, x_test / IMAGE_SIZE
#one-hot 
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

#We define or NN with 3 convolutional layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    #Sale un array de 10D, Softmax hace una distribucion de prob
    tf.keras.layers.Dense(10, activation='softmax')])
model.summary()

optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)

model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
tensorboard_callback = tf.keras.callbacks.TensorBoard(PATH)
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
         callbacks= [scheduler_callback, tensorboard_callback]) 
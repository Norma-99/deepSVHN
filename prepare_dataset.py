import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


TRAIN_SAMPLES = 70000
TEST_SAMPLES = 10000


def load_samples(samples, split):
    # Load dataset into a list
    dataset = tfds.load(name="svhn_cropped", split=split)
    dataset_list = list(tfds.as_numpy(dataset))

    # Split inputs and outputs
    x, y = list(), list()
    for pair in dataset_list[:samples]:
        x.append(pair['image'])
        y.append(pair['label'])

    # Change data format
    x = np.array(x) / 255
    y = tf.keras.utils.to_categorical(np.array(y))

    return x, y

if __name__ == '__main__':
    # Load training and test data
    x_train, y_train = load_samples(TRAIN_SAMPLES, split=tfds.Split.TRAIN)
    x_test, y_test = load_samples(TEST_SAMPLES, split=tfds.Split.TEST)

    # Put it into a single object
    save_target = (x_train, y_train), (x_test, y_test)

    # Pickle it to a file
    filename = 'dataset_train{train_samples}_test{test_samples}.pickle'.format(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES
    )
    with open(os.path.join('datasets', filename), 'wb') as f:
        pickle.dump(save_target, f)a
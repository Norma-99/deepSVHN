import pickle
import tensorflow as tf
from datetime import datetime


DATASET_PATH = 'datasets/dataset_train70000_test10000.pickle'
LEARNING_RATE = 0.1
EPOCHS = 20


(x_train, y_train), (x_test, y_test) = pickle.load(open(DATASET_PATH, 'rb'))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')])


optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

log_path = 'logs/experiment2_{unique_id}'.format(
    unique_id=int(datetime.now().timestamp()) 
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback])
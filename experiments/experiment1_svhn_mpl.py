import tensorflow as tf
from datetime import datetime


DATASET_PATH = 'dataset'
LEARNING_RATE = 0.1




model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

log_path = 'logs/experiment1_{unique_id}'.format(
    unique_id=int(datetime.now().timestamp()) 
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

model.fit()
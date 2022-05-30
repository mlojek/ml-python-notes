import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# load cifar (like mnist):
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        # output shape 30 30 32
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 2,2 is default pool size
        # output shape 15 15 32
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.0003),
    metrics=['accuracy'],
)

model.summary()

model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test, y_test)


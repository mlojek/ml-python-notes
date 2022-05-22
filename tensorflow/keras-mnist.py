import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import handwritten digits dataset
from tensorflow.keras.datasets import mnist

# get the data:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape input data:
# leave the first dimension unchanged, then flatten (28, 28) to (784):
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# optionally change f64 to f32 to ease computation:
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# normalize the values (is it necessary tho?):
x_train = x_train / 255.0
x_test = x_test / 255.0

# make a sequential model:
model = keras.Sequential(
    [
        keras.Input(shape=(784)),
        layers.Dense(512, activation=keras.activations.relu),
        layers.Dense(256, activation=keras.activations.relu),
        layers.Dense(10, activation=keras.activations.softmax),
    ]
)

# specify other aspects of the model:
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

# get info about net structure:
model.summary()

# train the net:
model.fit(x_train, y_train, epochs=100)

# evaluate the net:
model.evaluate(x_test, y_test)
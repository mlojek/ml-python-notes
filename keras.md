# Keras
https://keras.io/api/

## Neural networks:
```python
import tensorflow as tf
from tensorflow import keras

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
```

## Creating a neural network:
There are 3 approaches for creating a neural network in keras:
### 1. defined structure
```python
# define the structure of the model:
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        keras.layers.Dense(512, activation='relu', name='first layer'),
        keras.layers.Dense(256, activation='relu', name='second layer'),
        keras.layers.Dense(10),
    ]
)

# training setup:
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)
```

### 2. layer by layer
A model can also be built layer by layer like this:
```python
model = keras.Sequential()

# add a layer:
model.add(layer)

# remove the last layer:
model.pop()
```

### 3. functional api
Every layer is a function, that stores the previous layer.
Works kinda recursive.

```python
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Then compile and do stuff
```
## training

```python
# get info about net structure:
model.summary()

# train the net:
# x_train - inputs
# y_train - answers
# Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
# shuffle=True/False
# batch_size default 32
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)

# evaluate the net:
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
```


## other things
### activation functions
- default - no activation function
### loss functions

### optimizers:
TL;DR use Adam for the best performance  
  
- SGD
- RMSprop
- Adam
- Adadelta
- Adagrad
- Adamax
- Nadam
- Ftrl



## Get layer's outputs:
model.layers[-1] - get last layer
model.layers[0] - get first layer

outputs for every layer:
to get predictions do layer_outputs[-1]
layer_outputs = model.predict(x_test)

layer_object.output - access layer's outputs
model.get_layer('name')

model.predict_classes

## Saving/reading a model to/from a file:
HDF5 (.h5) is a file format for storing objects like neural networks:
```python
# save model to file:
model.save(path.h5)

# read model from file:
model = keras.models.load_model(path.h5)
```









## Convolutional networks:
### CIFAR-10
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
```
cifar 10 contains 50K + 10K 32x32 rgb images belonging to 10 different classes (like cat, dog etc)

```python
# load cifar (like mnist)
# three channels, so shape of input is (32, 32, 3)

layers.Conv2D(int:output, kernel_size:int, padding='valid'(default)/'same')
```
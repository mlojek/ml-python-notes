# Keras
https://keras.io/api/

```python
import tensorflow as tf
from tensorflow import keras
```

## Creating a neural network:
There are 3 approaches for creating a neural network in keras:

### 1. defined structure
```python
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        keras.layers.Dense(512, activation=keras.activations.relu, name='first layer'),
        keras.layers.Dense(256, activation=keras.activations.relu, name='second layer'),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ]
)
```

### 2. layer by layer
```python
model = keras.Sequential()

# add a layer:
model.add(keras.Input(shape=(28*28)))

# remove the last layer:
model.pop()
```

### 3. functional api
```python
input_layer = keras.Input(shape=(784))
x = keras.layers.Dense(512, activation=keras.activations.relu)(input_layer)
x = keras.layers.Dense(256, activation=keras.activations.relu)(x)
output_layer = keras.layers.Dense(10, activation=keras.activations.softmax)(x)

model = keras.Model(input_layer, output_layer)
```

### activation functions:
TL;DR:
- ALWAYS use an activation function
- for hidden layers use relu for fastest learning
- for output layer use:
    - no activation if approximating a function
    - tanh/sigmoid if output 0/1
    - softmax for probability distribution (like mnist)

FULL list:
- default - no activation function
- relu
- sigmoid
- softmax - probability distribution
- softplus
- softsign
- tanh
- selu
- elu
- exponential


## Compile stage:
```python
# training setup:
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

```
### loss functions:
PROBABILISTIC:
| function name          | net output           | true output         |
|------------------------|-----------------------------|-------------------------|
| BinaryCrossentropy     | one float between 0 and 1   | one int 0/1             |
| CategoricalCrossentropy | 
| SparseCategoricalCrossentropy | 
| Poisson
| KLDivergence

Regression:

MeanSquaredError class
MeanAbsoluteError class
MeanAbsolutePercentageError class
MeanSquaredLogarithmicError class
CosineSimilarity class
Huber
LogCosh

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


## Running the net:
```python
# print info about net structure:
model.summary()

# train the net:
# batch size
model.fit(x_train, y_train, epochs=10)

# evaluate the net:
model.evaluate(x_test, y_test)
```


## Get net's outputs:
```python
# get net outputs for every test case:
outputs = model.predict(x_test)
```

## Saving/reading a model to/from a file:
HDF5 (.h5) is a file format for storing objects like neural networks:
```python
# save model to file:
model.save(path.h5)

# read model from file:
model = keras.models.load_model(path.h5)
```


## Mnist dataset:

```python

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
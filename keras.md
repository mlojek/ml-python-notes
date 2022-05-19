# Keras

## Neural networks:
```python
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
```

## Creating a sequential model (keras sequential API):
```python
# make a sequential model:
model = keras.Sequential(
    [
        keras.Input(shape=tuple),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

# specify other aspects of the model:
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

# get info about net structure:
model.summary()

# train the net:
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)

# evaluate the net:
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
```

### Build the model layer by layer:
```python
model = keras.Sequential()
model.add(layer)
```
## Creating a functional model (keras functional API):
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

layers can be named with name='str'

### Get layer's outputs:
model.layers[-1] - get last layer
model.layers[0] - get first layer

outputs for every layer:
to get predictions do layer_outputs[-1]
layer_outputs = model.predict(x_test)

layer_object.output - access layer's outputs
model.get_layer('name')

### saving/reading file from/ti file:
model_obj.save(path)
model = keras.models.load_model(path)

### optimizers:
adam
gradient descent
RESEARCH!
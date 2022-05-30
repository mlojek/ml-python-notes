# Keras convolutional neural networks
computer vision  
https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
watch first 11 episodes before attempting to understand this document

## Layers:
```python
# flatten a multi-dimensional input to a linear output:
keras.layers.Flatten()

# 2D convolutional layer:
# filters - no of filters/output channels
# padding - "valid"==no padding, "same"==when strides=1 output same size as input
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    activation='relu'
) 

# Max/Average Pooling layer:
# Padding same means equal size input and output
# default (2, 2), (2, 2) - output 2 times smaller in both dimensions
tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=None,
    padding="valid"
)
```

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

model = keras.Sequential{
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
        layers.Dense(10),
    ]
}

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.0003),
    metrics=['accuracy'],
)

model.fit batch 64, 10 epochs
model.evaluate batch 64



```
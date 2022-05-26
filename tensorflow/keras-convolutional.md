# Keras convolutional neural networks
computer vision  
https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
watch before attempting to understand this document

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

# layers.Conv2D(output_channesl, kernel_size(height, width) or just int, padding='valid'(default)/'same')

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
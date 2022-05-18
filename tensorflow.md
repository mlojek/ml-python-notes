# tensorflow
```python
import tensorflow as tf

# to silence annoying tf logs:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


## Tensors:
a tensor is...

indexed like a list, [:::]
tf.gather(x, [indices])

### Initializing tensors:
scalar tensor of constant value
x = tf.constant(int/list, shape=tuple, dtype=type)
x = tf.constant(4)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
types:
tf.float32/float64/int32/int64/bool
float32 is the most common
also 8 and 16 exist, and float16 is sometimes used

x = tf.ones(tuple)
x = tf.zeros
x = tf.eye(int) - identity matrix (1 on diagonal, 0 rest)
in all above you can use dtype= as well

x = tf.random.normal(tuple, mean=float, stddev=float)
x = tf.random.uniform(tuple, minval=float, maxval=float)
x = tf.range(int) - [0, 1 ... int]
x = tf.range(start=float, limit=int, delta=int)
delta is the step

cast vector to type:
x = tf.cast(x, dtype)
x = x.astype('type')

### Mathematical operations on tensors: 
```python
# add/subtract/multiply/divide element-wise (respective elements from each tensor):
z = tf.add(x, y)
z = x + y

z = tf.subtract(x, y)
z = x - y

z = tf.multiply(x, y)
z = x * y

z = tf.divide(x, y)
z = x / y

# exponentiate all elements:
z = x ** 2

# dot product (sum of element-wise multiplication):
z = tf.tensordot(x, y, axes=1)

# other way of calculating dot product:
z = tf.reduce_sum(x*y, axis=0)

# matrix multiplication:
z = tf.matmul(x, y)
z = x @ y

```

### Indexing a tensor:
like an array

### Reshaping a tensor:
```python
x = tf.reshape(x, tuple)

# transposing a tensor (works for more > 2 dimensions as well):
x = tf.transpose(x, perm=[1, 0])
perm - new order of the indices
```

## Neural networks:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

```python
# import handwritten digits dataset
from tensorflow.keras.datasets import mnist

# get the data:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape input data:
# leave the first dimension unchanged, then flatten (28, 28) to (784):
x_train = x_train.reshape(-1, 784)
# optionally change f64 to f32 to ease computation:
x_train = x_train.astype("float32")
# normalize the values:
x_train = x_train / 255.0
```
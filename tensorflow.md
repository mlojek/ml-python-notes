# tensorflow
Great tutorial playlist I used: https://youtube.com/playlist?list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb

```python
import tensorflow as tf

# to silence annoying tf logs:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# in case of any errors:
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Tensors:
A tensor is an n-dimensional array stored in and handled by the GPU. It can be addressed like a python list with [ ], slicing with [::] also works.

## Data types (dtype):
`tf.float8/16/32/64`  
`tf.int8/16/32/64`  
`tf.bool`  
  
It's best to use float16/32 for the best performance.

## Creating tensors:
```python
# initialize a tensor:
x = tf.constant(int/list, shape=tuple, dtype=type)
x = tf.constant(4, shape=(1), dtype=tf.float16)
x = tf.constant([[1, 2, 3], [4, 5, 6]])

# homogenous tensors:
x = tf.ones(tuple, dtype=type)
x = tf.zeros(tuple, dtype=type)

# identity matrix (1 on the diagonal, 0 everywhere else):
x = tf.eye(int)

# tensors of random values:
x = tf.random.normal(tuple, mean=float, stddev=float)
x = tf.random.uniform(tuple, minval=float, maxval=float)

# tensor of a range (0 to int-1):
x = tf.range(int)

# other range (start to limit-1, step of delta):
x = tf.range(start=int, limit=int, delta=int)
```

## Mathematical operations on tensors: 
```python
# add/subtract/multiply/divide element-wise (respective elements from each tensor):
x = tf.add(x, y)
x = tf.subtract(x, y)
x = tf.multiply(x, y)
x = tf.divide(x, y)

x = x + y
x = x - y
x = x * y
x = x / y

# exponentiate all elements:
x = x ** 2

# dot product (sum of element-wise multiplication):
x = tf.tensordot(x, y, axes=1)

# other way of calculating dot product:
x = tf.reduce_sum(x*y, axis=0)

# matrix multiplication:
x = tf.matmul(x, y)
x = x @ y

```

## Manipulating a tensor:
```python
# reshape:
x = tf.reshape(x, tuple)

# transposing a tensor (works for more > 2 dimensions as well)
# perm is the new order of dimensions:
x = tf.transpose(x, perm=[1, 0])

# cast to another type:
x = tf.cast(x, dtype)
x = x.astype('dtype')
```

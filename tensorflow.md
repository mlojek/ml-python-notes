# tensorflow
https://youtube.com/playlist?list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb

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

Indexing a tensor: like an array
indexed like a list, [::]
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

### Reshaping a tensor:
```python
x = tf.reshape(x, tuple)

# transposing a tensor (works for more > 2 dimensions as well):
x = tf.transpose(x, perm=[1, 0])
perm - new order of the indices
```

# tensorflow
```python
import tensorflow as tf

# to silence annoying tf logs:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
## Tensors:
a tensor is...

### Initializing tensors:
scalar tensor of constant value
x = tf.constant(int/list, shape=tuple, dtype=type)
x = tf.constant(4)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
types:
tf.float32/float64/int32/int64

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



### Mathematical operations on a tensor:

### Indexing a tensor:

### Reshaping a tensor:
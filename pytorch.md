# pyTorch
https://www.youtube.com/watch?v=c36lUUr864M

```
pip install torch torchvision torchaudio 
on manjaro do also:
pamac install python-pytorch-cuda
```

```python
import torch
```

x = torch.rand(3) - random tensor
torch.cuda.is_available()

## tensors:
creating a tensor:
```python
x = torch.empty(1)
x = torch.empty(2, 3) - not a tuple!!!
torch.zeros, ones
x.dtype
default float32
torch.rand(3, dtype=)
torch.int .float16 etc
x.size()
x = torch.tensor(list)

torch.cuda.Tensor()
```
element-wise operations:
```python
elementwise addition
+ or torch.add
or x.add_(y) - inplace
all leading underline methods are inplace
same for - * /
.sub .mul .div

+= -= *= /= all elements
```
slicing supported

x[1,1] returns tensor object
x[1,1].item() returns value

resize a tensor:
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(2, 2, 4)
a = x.view(-1, 8) - will figure out on its own

x = torch.from_numpy(ndarray)

cuda  
put tensor on gpu
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)

    x = torch.ones(5)
    y = x.to(device)
    to('cpu')

    torch.ones(5, requires_grad=True)
```

# autograd
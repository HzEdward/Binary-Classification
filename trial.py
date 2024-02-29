#wirte a use of the function torch_max
import torch
import numpy as np
a = torch.tensor([1,2,3,4,5,6,7,8,9,10])
b = torch.tensor([10,9,8,7,6,5,4,3,2,1])
print(torch.max(a,b))


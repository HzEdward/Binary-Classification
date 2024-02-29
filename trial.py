#wirte a use of the function torch_max
import torch
import numpy as np
a = torch.tensor([[ 2.4193, -2.5289],
        [-3.0946,  2.7564],
        [ 2.5607, -2.7465],
        [ 2.6169, -2.6538],
        [-3.1493,  3.3492],
        [-2.4661,  3.0327]])
# torch_max is used to compare two tensors and return the maximum value of each element
print(torch.max(input=a,dim=1, keepdim=False, out=None))   


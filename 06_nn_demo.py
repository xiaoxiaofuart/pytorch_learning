"""
    pytorch 中可以自定义一个神经网络，通过实现nn.Module实现，重写其中的__init__与forward函数
"""
import torch
from   torch import nn
class MyFirstNN(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input+1
        return output



if __name__ == '__main__':
   myFirstNN = MyFirstNN()
   tensor_instance = torch.tensor(1)
   output = myFirstNN(tensor_instance)
   print(output)
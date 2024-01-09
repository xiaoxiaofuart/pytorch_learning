"""
    pytorch中可以使用dataloader指定dataset的加载方式，方法中包含几个参数
        1)第一个参数表示传递的数据集
        2）第二个参数表示一次拉取的数据量
        3）第三个表示是否乱序抓取，如果选是，那么下一次运行的时候图片顺序会和本次不通
        4）drop_last表示最后一次可能取到的数量与batch_size不相等，是否删除
"""

import  torch.utils.data.dataloader
import torchvision
import torch
from  torch.utils.tensorboard  import SummaryWriter

test_datasets = torchvision.datasets.CIFAR10("./datasets",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True,drop_last=False)


writer = SummaryWriter("dataloaders")
step = 0
for dataitem in dataloader:
    imgs, targets = dataitem
    writer.add_images("dataloaders", imgs, step)
    step = step+1

writer.close()
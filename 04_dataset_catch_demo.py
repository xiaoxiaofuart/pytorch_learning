"""
    pytorch官网给了很多案例用于获取数据集，CIFAR10是图像类别数据集，可以使用torchvision.datasets.CIFAR10函来创建一个数据集
        1）第一个参数表示数据集下载后放置的位置
        2）第二个参数表示是否是训练数据
        3）第三个参数表示使用的transform函数
        4）第四个表示是否需要下载数据集，默认为True即可
    函数返回的dataset是一个元素，元素第一个数据表示数据格式，比如现在获取到的是Image，第二个数据表示的是目标数据位置，通过tesorboard可以将这些数据展示出来
"""


import torchvision
from  torch.utils.tensorboard  import  SummaryWriter

tesner_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset_train = torchvision.datasets.CIFAR10("./datasets", train=True,transform=tesner_trans, download=True)
dataset_test = torchvision.datasets.CIFAR10("./datasets", train=False,transform=tesner_trans, download=True)

print(dataset_train[0])
# dataset_train[0][0].show()

writer = SummaryWriter("trains")

for i in range(10):
    img, target = dataset_train[i]
    writer.add_image("trains", img, i)

writer.close()

"""
    pytorch 中transform的作用是用来对 数据进行处理转换 比如 Image ->Tensor; ndarry-> Tensor,图像进行等比方法所辖，随机截取等操作
    1）transform实际上是一个py类，里面定义了许多类，比如ToPIT ，ToTensor 等，使用这类函数时关注__call__方法，查看输入输出
    2）另外transform 中提供了compose类可以定义一系列的转换操作进行处理
"""


# 读取图像，有多种方式，可以使用PIL，或者opencv或者pyplib等，读到的数据返回对象不一样
import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

image = cv2.imread("hymenoptera_data/train/bees/16838648_415acd9e3f.jpg")

print(image.shape)

image_tensor = transforms.ToTensor()

instance_tensor = image_tensor(image)

writer = SummaryWriter("tensors")

writer.add_image("transform",instance_tensor)


# 图像归一化处理
image_normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

normalize_tensor = image_normalize.forward(instance_tensor)

writer.add_image("transform_1",normalize_tensor)

#图像的resize操作

resize = transforms.Resize([200,200])
resize_tensor = resize.forward(instance_tensor)
writer.add_image("transform_2",resize_tensor)



# 使用compose，进行多个操作链式处理
image_tensor1 = transforms.ToTensor()
compose = transforms.Compose([transforms.Resize([900,1000]),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
compose_sensor = compose(image_tensor1(image))
writer.add_image("transform_3",compose_sensor)

writer.close()
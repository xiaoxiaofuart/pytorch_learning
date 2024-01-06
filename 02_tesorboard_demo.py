"""
    tensorborad 可以用于绘制模型训练中损失函数，但是需要注意目前tesorflow只支持3.7-3.9的python，如果安装了高版本python，需要进行降级
    通过SummaryWriter，初始化一个绘制类，构造方法中可传递日志生成路径
    使用add_scalar添加一个标量图，第一个参数表示标量图的名称，第二个参数表示y轴，第三个参数表示x轴
    在终端命令行中输入 tensorboard --logdir=事件相对路径  --port=6006 来启动一个人board，运行成功会显示TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)在浏览器中打开即可
    add_image用于添加图片到变更事件中，第一个参数表示事件名字，第二个参数表示图片数据，第三个参数表示步骤，dataformates用于指定数据格式，默认是CHW
"""

from torch.utils.tensorboard import SummaryWriter
import  numpy as np
import  matplotlib.pyplot as plt

# 指定标量日志文件的生成位置
obj_writer = SummaryWriter("logs")

# 读取文件
obj_image = plt.imread("hymenoptera_data/train/ants/0013035.jpg")

# 输出图片
obj_writer.add_image("my_image",obj_image,1,dataformats="HWC")


# obj_writer.add_image()
# 添加一个标量，用于绘制标量图
for i in range(100):
    obj_writer.add_scalar("y==x",2*i,i)

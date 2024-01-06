'''
    pytorch中有两个重要概念，分别是dataset与datasetLoder
    dataset是用来表示数据集，所有数据集都要继承torch.utils.data.Dataset，他有两个方法，__get_item__用于获取数据集，__len__表示数据集的长度
'''
import os.path

from torch.utils.data import Dataset
from PIL import Image


# 定义一个类继承自Dataset，实现__getitem__方法，返回一个Image对象
class MyFirstDataset(Dataset):

    def __init__(self, baseDir: str, labelDir: str) -> None:
        self.baseDir = baseDir
        self.labelDir = labelDir
        path_dir = os.path.join(self.baseDir, self.labelDir)
        self.images = os.listdir(path_dir)

    def __getitem__(self, index) -> Image:
        images_name = self.images[index]
        image_obj = Image.open(os.path.join(self.baseDir, self.labelDir, images_name))
        return image_obj

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    obj_dataset = MyFirstDataset("hymenoptera_data/train", "ants")
    obj_image = obj_dataset.__getitem__(0)
    # obj_image.show()
    print(f"The ants dataset lenght is {obj_dataset.__len__()}")
    bees_dataset = MyFirstDataset("hymenoptera_data/train", "bees")
    print(f"The bees dataset lenght is {bees_dataset.__len__()}")
    # 可以对两个dataset进行相加，得到新的数据集
    union_dataset = obj_dataset + bees_dataset
    print(f"The union dataset lenght is {union_dataset.__len__()}")
    union_dataset[244].show()

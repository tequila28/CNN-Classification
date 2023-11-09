import torch
from torchvision import datasets
from torchvision import transforms

# 导入训练集数据
from torch.utils.data import dataloader

train_data =  datasets.CIFAR10(root='C:/Users/86159/PycharmProjects/Resnet图像分类/data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),  # 将图片转化为tensor

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]), download=True)


# 导入测试集数据
test_data = datasets.CIFAR10(root='C:/Users/86159/PycharmProjects/Resnet图像分类/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]), download=True)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

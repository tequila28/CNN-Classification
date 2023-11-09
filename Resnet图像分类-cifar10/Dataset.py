
from torchvision import datasets
from torchvision import transforms

# 导入训练集数据
from torch.utils.data import dataloader
myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_data =  datasets.CIFAR10(root='data/', train=True, transform=myTransforms)


# 导入测试集数据
test_data = datasets.CIFAR10(root='data/', train=False, transform=myTransforms)

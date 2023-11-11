import torchvision
import torchvision.transforms as transforms
import  torch
import torch.nn as nn
import torch.optim as optim
from ViT import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
trainset=torch.utils.data.DataLoader(train_data,batch_size=100, shuffle=True)
testset=torch.utils.data.DataLoader(train_data,batch_size=100, shuffle=True)

net=ViT(32,4,10,32,8,4).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainset, 0):
        inputs, labels = data
        inputs, labels=inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    with open("vit_training_records.txt", "a") as f:
        f.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainset)}\n')
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainset)}')

print("Finished Training")
# 保存和加载模型
torch.save(net.state_dict(), 'cifar10_vit2.pth')
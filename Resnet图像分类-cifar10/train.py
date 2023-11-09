import  time
import torch.optim as optim
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from  Resnet_50 import resnet18
from Dataset import train_data
from test import test_net

def train_net(net, device, epochs=10, batch_size=50, lr=0.01):

    # 加载数据集
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    # 定义损失函数和优化器
    #optimizer = optim.SGD(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(),lr=1e-1, momentum=0.9, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss()
    writer = SummaryWriter("loss")
    total_train_loss = 0
    # 训练模型
    for epoch in range(epochs):
        for i, (inputs,labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, batch {i+1} loss: {loss.item()} ')
            # 保存模型
            if (i+1)%100==0:
                print(f'Epoch {epoch + 1}, batch {i+1}, average loss: {total_train_loss/100} ')
                writer.add_scalar("train_loss", total_train_loss/100, i+epoch* 1000)
                total_train_loss = 0

    writer.close()



if __name__ == '__main__':
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型并移动到设备

    start = time.time()
    net = resnet18().to(device)


    #net=torch.load("network.pth")

    train_net(net, device)
    torch.save(net,'network2.pth')
    end=time.time()
    timeall=end-start
    print(f"训练时间:{timeall:.2f}秒")

    # 训练模型

    test_net(net,device)

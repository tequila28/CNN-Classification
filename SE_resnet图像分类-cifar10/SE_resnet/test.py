from torch import nn
import torch
from torch.utils.data import DataLoader
from dataset import test_data



def test_net(net, device, epochs=1, batch_size=100):
    net.eval()
    with torch.no_grad():
      test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # 定义损失函数和优化器
      criterion = nn.CrossEntropyLoss()

    # 训练模型
      for epoch in range(epochs):
          correct = 0
          total = 0
          for i, data in enumerate(test_loader, 0):
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = net(inputs)
              loss = criterion(outputs, labels)
              predicted = outputs.argmax(dim=1)
              total += labels.size(0)
              #print(labels)
              #print(predicted)
              correct += torch.eq(predicted, labels).float().sum().item()
              print(f'Epoch {epoch + 1}, batch {i + 1} loss: {loss.item()} ')
          print('Accuracy of the network on the 10000 test images: %d %%' % (
                  100 * correct / total))




            # 保存模型


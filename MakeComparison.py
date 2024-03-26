import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import MyDataLoader
import numpy as np
from Models import TimeSeriesPredictor
from Utils import load_checkpoint


def Comparison():
    # 定义一些超参数
    input_dim = 50  # 输入数据的特征维度
    batch_size = 256  # 批次大小
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 打印设备信息
    print(device)
    # 构建模型
    model = TimeSeriesPredictor(input_size=input_dim, hidden_size=128, device=device).to(device)  # 注意将模型也转移到GPU上
    # 移动模型到设备上
    # model.to(device)
    # 接着，你可以用下面的代码来定义损失函数和优化器：
    # 定义均方误差损失函数
    criterion = nn.MSELoss()
    # 定义随机梯度下降优化器
    optimizer = torch.optim.Adadelta(model.parameters())
    # checkpoint = torch.load('checkpoint.pth')
    checkpoint = torch.load('best_val_checkpoint.pth')
    model, optimizer, start_epoch, start_loss, best_val_loss = load_checkpoint(checkpoint, model, optimizer)
    print(f'Epoch {start_epoch}, Loss: {start_loss:.4f}')

    # 进入评估模式
    model.eval()
    # 初始化测试损失为零
    test_loss = 0.0
    # 初始化预测值和真实值的列表为空
    preds_np = []
    truths_np = []
    # 然后，构建并加载数据
    test_loader = MyDataLoader.build_loader(batch_size, train_mode=False)

    # 遍历测试数据集中的每个批次
    for inputs, labels in test_loader:
        # 将输入和标签转换为张量，并移动到设备上
        inputs = inputs.view(-1, 120, input_dim).to(device)
        labels = labels.view(-1, 1).to(device)
        # 前向传播，得到预测值
        outputs = model(inputs)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 累加测试损失值
        test_loss += loss.item()
        # 将预测值和真实值添加到列表中
        preds_np.append(outputs.detach().cpu().numpy())
        truths_np.append(labels.detach().cpu().numpy())
    # 计算并打印平均测试损失值
    test_loss = test_loss / len(test_loader)
    print('Test Loss: %.4f' % test_loss)

    preds = []
    truths = []

    for i in range(len(preds_np)):
        for j in range(len(preds_np[i])):
            preds.append(preds_np[i][j])
            truths.append(truths_np[i][j])

    # 将预测值和真实值的列表转换为数组
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    mean = 2.493994332513462
    std = 4.065262472229068
    # 逐图绘制预测结果
    with open('Comparison.txt', 'w') as f:
        f.write('Preds:\tTruths:\n')
        for i in range(len(truths)):
            print(f'{i + 1}//{len(truths)}')
            plt.scatter(i, (preds[i] * std) + mean, c='red')
            plt.scatter(i, (truths[i] * std) + mean, c='green')
            f.write(f'{((preds[i] * std) + mean):.4f}\t, {((truths[i] * std) + mean):.4f}\n')
            pass
        plt.savefig('Comparison.png')

if __name__ == '__main__':
    Comparison()

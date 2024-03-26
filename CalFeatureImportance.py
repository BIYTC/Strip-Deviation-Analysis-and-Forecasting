import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import MyDataLoader
import numpy as np
from Models import TimeSeriesPredictor
from Utils import load_checkpoint
import seaborn as sns


def calculate_row(A, m):
    # 计算第m行之和
    row_sum = np.sum(A[m])
    # 计算第m行第m列的数值
    self_value = A[m][m]
    # 计算第m列的熵
    colum_entropy = -np.sum(A[:,m] * np.log2(A[:,m]))
    # 计算第m行的结果
    S_m = row_sum * self_value / colum_entropy
    # 返回结果
    # return S_m
    return row_sum



def attentionHot():
    # 定义一些超参数
    input_dim = 40  # 输入数据的特征维度
    batch_size = 1  # 批次大小
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 打印设备信息
    print(device)
    # 实例化模型，并将其转移到GPU上
    model = TimeSeriesPredictor(input_size=input_dim, hidden_size=64, device=device).to(device)  # 注意将模型也转移到GPU上
    # 接着，你可以用下面的代码来定义损失函数和优化器：
    # 定义均方误差损失函数
    criterion = nn.MSELoss()
    # 定义随机梯度下降优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adadelta(model.parameters())
    checkpoint = torch.load('best_val_checkpoint.pth')
    model, optimizer, start_epoch, start_loss, _ = load_checkpoint(checkpoint, model, optimizer)
    print(f'Epoch {start_epoch}, Loss: {start_loss:.4f}')

    # 进入评估模式
    model.eval()

    # 然后，构建并加载数据
    test_loader = MyDataLoader.build_loader(batch_size, train_mode=False)

    # 遍历测试数据集中的每个批次
    for i, (inputs, labels) in enumerate(test_loader):
        if i == 77:
            # 将输入和标签转换为张量，并移动到设备上
            inputs = inputs[:, :, :-1].view(-1, 120, input_dim).to(device)
            labels = labels.view(-1, 1).to(device)
            # 前向传播，得到预测值
            outputs, b, c = model(inputs)
            t_array = b.detach().cpu().numpy().squeeze()  # 去掉第一维度为1的维度，转换为numpy数组
            np.savetxt("FeatureAttention.txt", b.squeeze().detach().cpu().numpy(), fmt='%f',delimiter=',')
            # 获取数组的维度
            n = t_array.shape[0]
            # 遍历每一行
            for j in range(n):
                # 调用函数计算结果
                result = calculate_row(t_array, j)
                # 打印结果
                print(f"第{j + 1}行的结果是:{result:.5f}")
            break

if __name__ == '__main__':
    attentionHot()

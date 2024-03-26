import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import MyDataLoader
import numpy as np
from Models import TimeSeriesPredictor
from Utils import save_checkpoint, load_checkpoint
from torch.cuda.amp import autocast, GradScaler


# 直接用MAE用评价模型好坏，不用MSE了
def val(model, batch_size, input_dim, device, criterion, draw=False):
    print("=> evaling")
    # 进入评估模式
    model.eval()
    # 初始化测试损失为零
    test_loss = 0.0
    # 初始化预测值和真实值的列表为空
    preds = []
    truths = []
    # 然后，构建并加载数据
    test_loader = MyDataLoader.build_loader(batch_size, False)
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
        preds.append(outputs.detach().cpu().numpy())
        truths.append(labels.detach().cpu().numpy())

    # 将预测值和真实值的列表转换为数组
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    mean = 2.493994332513462
    std = 4.065262472229068
    mae = np.mean(np.abs((truths * std + mean) - (preds * std + mean)))
    print('Test Loss: %.4f' % mae)

    '''
    # 计算并打印平均测试损失值
    test_loss = test_loss / len(test_loader)
    print('Test Loss: %.4f' % test_loss)

    # 将预测值和真实值的列表转换为数组
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)

    if draw:
        # 绘制预测值和真实值的散点图，并显示相关系数
        plt.scatter(truths, preds)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Correlation: %.4f' % np.corrcoef(truths.ravel(), preds.ravel())[0, 1])
        plt.show()
        return test_loss
    '''
    return mae


def train():
    # 定义一些超参数
    input_dim = 50  # 输入数据的特征维度
    batch_size = 128  # 批次大小
    val_batch_size = 16
    num_epochs = 250  # 定义训练轮数
    print_step = 1  # 定义每轮打印的步数并保存模型
    val_step = 1  # 定义验证轮数
    # 定义一个标志变量，表示是否需要恢复训练，你可以根据你的需要来修改这个变量的值
    # resume_training = False
    resume_training = True
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 打印设备信息
    print(device)
    # 构建模型
    # 实例化模型，并将其转移到GPU上
    model = TimeSeriesPredictor(input_size=input_dim, hidden_size=128, device=device).to(device)  # 注意将模型也转移到GPU上
    # 移动模型到设备上
    model.to(device)
    # 接着，你可以用下面的代码来定义损失函数和优化器：
    # 定义均方误差损失函数
    criterion = nn.MSELoss()
    # 定义随机梯度下降优化器
    optimizer = torch.optim.Adadelta(model.parameters())
    scaler = GradScaler()
    best_val_loss = 31415926535
    # 是否继续训练
    if resume_training:
        checkpoint = torch.load('checkpoint.pth')
        model, optimizer, start_epoch, start_loss, best_val_loss = load_checkpoint(checkpoint, model, optimizer)
        print(f'Resume from Epoch {start_epoch + 1}, Loss: {start_loss:.4f}, best_val_loss: {best_val_loss}')
        write_mode = 'a'
    else:
        print('No Resume!')
        start_epoch = 0
        write_mode = 'w'
    # 然后，构建并加载数据
    train_loader = MyDataLoader.build_loader(batch_size)

    # 进入训练模式
    model.train()
    # 遍历训练轮数
    for epoch in range(start_epoch, num_epochs):
        # 初始化累计损失为零
        epoch_loss = 0.0
        # 遍历训练数据集中的每个批次
        for i, (inputs, labels) in enumerate(train_loader):
            # 将输入和标签转换为张量，并移动到设备上
            inputs = inputs.view(-1, 120, input_dim).to(device)
            labels = labels.view(-1, 1).to(device)

            # 清零梯度缓存
            optimizer.zero_grad()
            with autocast():
                # 前向传播，得到预测值
                outputs = model(inputs)
                # 计算损失值
                loss = criterion(outputs, labels)
            # 缩放损失值
            scaler.scale(loss).backward()
            # 检查和处理梯度值
            scaler.unscale_(optimizer)
            # 更新参数
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()
            # 累加损失值
            epoch_loss += loss.item()
        # 每隔print_step个轮数打印一次当前的轮数和损失值，并保存一次模型状态字典
        if (epoch + 1) % print_step == 0:
            # 更新学习率
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')
            # 创建一个状态字典，包含模型参数、优化器状态、轮数和损失值
            state = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'best_val_loss': best_val_loss
            }
            # 保存状态字典到本地文件
            save_checkpoint(state, 'checkpoint.pth')
            # 保存loss
            with open(f'loss-batch{batch_size}.txt', write_mode) as f:
                f.write(f'Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}\n')
                write_mode = 'a'

        # 每隔val_step个轮数验证一次
        if (epoch + 1) % val_step == 0:
            val_loss = val(model, val_batch_size, input_dim, device, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 创建一个状态字典，包含模型参数、优化器状态、轮数和损失值
                state = {
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'best_val_loss': best_val_loss
                }
                # 保存最佳测试模型状态字典到本地文件
                save_checkpoint(state, 'best_val_checkpoint.pth')
                save_checkpoint(state, 'checkpoint.pth')
                print(f"=>Save best val checkpoint at {epoch} epoch")
            print(f'current val loss: {val_loss},best_val_loss: {best_val_loss:.4f}')
            # 保存best_val_loss
            with open(f'val_loss-batch{val_batch_size}.txt', write_mode) as f:
                f.write(f'Epoch：{epoch + 1}, Loss: {val_loss:.4f}\n')
                write_mode = 'a'
            # 进入训练模式
            model.train()

    _ = val(model, val_batch_size, input_dim, device, criterion, draw=True)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train()

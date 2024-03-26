import torch
import torchvision
from torchinfo import summary
from Models import TimeSeriesPredictor


# 定义一些超参数
input_dim = 40  # 输入数据的特征维度
batch_size = 128  # 批次大小
input = torch.randn(120, input_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 打印设备信息
print(device)
# 构建模型
model = TimeSeriesPredictor(input_size=input_dim, hidden_size=64, device=device).to(device)  # 注意将模型也转移到GPU上
print(summary(model=model, input_size=(1, 120, input_dim)))

import os
import torch
import pandas as pd
# from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 定义Standardize类
class Standardize(object):
    def __init__(self, mean, std):
        # mean: 数据的均值
        # std: 数据的标准差
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor: 一个时序数据张量，形状为[seq_len, feature_dim]
        # 返回一个标准化后的时序数据张量，形状不变
        return (tensor - self.mean) / self.std


# 定义Normalize类
class Normalize(object):
    def __init__(self, min_value, max_value):
        # min_value: 数据的最小值
        # max_value: 数据的最大值
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):
        # tensor: 一个时序数据张量，形状为[seq_len, feature_dim]
        # 返回一个归一化后的时序数据张量，形状不变
        return (tensor - self.min_value) / (self.max_value - self.min_value)


class MultiTransform(object):
    def __init__(self, transforms):
        # transforms: 一个字典，键为属性名，值为转换函数
        self.transforms = transforms

    def __call__(self, tensor, islabel=False):
        # tensor: 一个时序数据张量，形状为[seq_len, feature_dim]
        # 返回一个按照不同属性进行不同转换后的时序数据张量，形状不变
        # 遍历每个属性
        if islabel:
            transform = self.transforms[2]  # diff3所对应的属性!!!!!!!!!!!!!!!!!!
            # 对该属性进行转换，并替换原始数据
            tensor = transform(tensor)
            return tensor
        else:
            for i in range(len(self.transforms)):
                # 获取对应的转换函数
                transform = self.transforms[i]
                # 对该属性进行转换，并替换原始数据
                tensor[:, i] = transform(tensor[:, i])
            return tensor


class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # root_dir: 存放.pkl文件的文件夹路径
        # transform: 可选的数据转换函数
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有.pkl文件的文件名列表
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith('.pkl')]
        self.file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    def __len__(self):
        # 返回数据集的大小，即.pkl文件的数量
        return len(self.file_list)

    def __getitem__(self, idx):
        # 根据给定的索引，返回一个数据样本和其对应的标签
        # 读取对应的.pkl文件
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = pd.read_pickle(file_path)
        # 获取时序数据和标签
        sequence = data[0]
        label = data[1]
        # 将时序数据和标签转换为张量
        sequence = torch.from_numpy(sequence).float()
        label = torch.from_numpy(np.array(label)).float()
        # 如果有转换函数，就对数据进行转换
        if self.transform:
            sequence = self.transform(sequence, islabel=False)
            label = self.transform(label, islabel=True)
        # 返回数据样本和标签
        return sequence, label


def build_loader(batch_size=32, train_mode=True):
    # 读取各个属性的平均值标准差文件
    df = pd.read_csv("avg_std.txt", header=None, sep="\t", encoding='gb18030')
    # 创建MultiTransform对象，假设第一个属性需要归一化，第二个属性需要标准化，第三个属性需要对数变换，第四个属性需要差分变换
    multi_transform = MultiTransform([
        Standardize(df.iloc[0, i], df.iloc[1, i]) for i in range(df.shape[1])
    ])

    if train_mode:
        # 创建训练所需的TimeSeriesDataset对象
        dataset = TimeSeriesDataset(root_dir='E:\\带钢跑偏PKL数据集\\JMPDataset\\train\\',
                                    transform=multi_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    else:
        # 创建验证和测试所需的TimeSeriesDataset对象
        dataset = TimeSeriesDataset(root_dir='E:\\带钢跑偏PKL数据集\\JMPDataset\\validation\\', transform=multi_transform)
        # dataset = TimeSeriesDataset(root_dir='E:\\带钢跑偏PKL数据集\\JMPDataset\\test\\', transform=multi_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return data_loader


if __name__ == '__main__':
    train_loader = build_loader()
    # 遍历训练集中的每个批次
    for inputs, labels in train_loader:
        # 打印输入和输出的形状
        print(inputs.shape, labels.shape)
        # 只打印一个批次，然后退出循环
        break

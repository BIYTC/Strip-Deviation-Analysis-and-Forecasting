import pandas as pd
import numpy as np
import pickle
import os


def average_time_series_with_pandas(data, s):
    # 计算每s秒需要平均的数据点数
    n = s * 10
    # 创建一个空的dataframe来存储平均后的数据
    avg_df = pd.DataFrame(columns=data.columns)
    # 遍历原始dataframe，每次跳过n个元素
    for i in range(0, len(data), n):
        # 取出当前位置到下一个n个元素之间的子dataframe
        sub_df = data.iloc[i:i + n, :]
        # 计算每一列的平均值，并存储在一个新的Series对象中
        avg_series = sub_df.mean(axis=0)
        # 将平均值Series对象添加到平均dataframe中
        avg_df = pd.concat([avg_df, avg_series.to_frame().T], ignore_index=True)
    # 返回处理后的dataframe
    return avg_df


def save_splited_data(filename, save_path, x, y, index):
    # 打开一个新的pkl文件，命名为00001_index.pkl
    f = open(save_path + filename.split('.')[0] + "_" + f"{index}" + ".pkl", "wb")
    # 将X和y保存到pkl文件中
    pickle.dump((x, y), f)
    # 关闭文件
    f.close()


def split_data(data, seq_len, pred_len, step, filename, save_path, avg=1):
    '''
    :param save_path:
    :param filename:
    :param data:
    :param seq_len:
    :param pred_len:
    :param step:
    :param avg: 将0.1s一个的数据点平均avg秒一个
    :return:
    '''
    # 获取数据框中除了time和空列之外的所有特征名称
    data = data.dropna(axis=1)
    features = data.columns.drop(["time", "Diff_Ave_R3", "Tilt_R3", "Diff_Ave_R5", "Tilt_R5"])
    data = average_time_series_with_pandas(data[features], avg)
    # 获取数据框中time和Diff_Ave_F3两个特征
    diff = data["Diff_Ave_F3"]
    # 遍历所有可能的起始时间点
    for i in range(0, len(data) - (seq_len + pred_len), step):
        # 获取该时间段内的所有特征值，并转换为numpy数组
        x = data[i:i + seq_len].to_numpy()
        # 获取该时间段结束时刻后m秒的Diff_Ave_F3值，并转换为numpy数组
        y = diff[i + seq_len + pred_len - 1]
        # 新增一个过滤机制来过滤掉等待期间的预测
        if y != x[-1, 2] and y != x[-2, 2] and y != x[-3, 2]:
            # 将x和y保存为一个.pkl文件
            save_splited_data(filename, save_path, x, y, index=i / step + 1)


def deal_txt_data_folds(path, save_path):
    # 创建一个名为"sequences"的文件夹
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 遍历文件夹中的所有文件
    for filename in os.listdir(path):
        # 检查文件名是否以.txt结尾
        if filename.endswith('.txt'):
            # 拼接文件路径
            print(f"正在处理{filename}\n")
            file_path = os.path.join(path, filename)
            # 读取文件
            df = pd.read_csv(file_path, sep="\t", header=1, encoding='gb18030')
            # 调用split_data函数，传入参数n=历史时间, m=预测时间, step=两样本间隔
            split_data(df, seq_len=120, pred_len=3, step=5, filename=filename, save_path=save_path)
            # split_data(df, n=120, m=3, step=1, filename=filename, save_path=save_path)  # 制作GIF用


if __name__ == '__main__':
    deal_txt_data_folds(path="E:\\LSTM数据集\\带跑偏量的_45个属性\\230710\\", save_path="D:\\Desktop\\days3_features41_step5_filtration\\")
    # deal_txt_data_folds(path="D:\\Desktop\\MakeGIF\\oridata\\00025\\",
    #                    save_path="D:\\Desktop\\MakeGIF\\dataset\\00025\\")  # GIF

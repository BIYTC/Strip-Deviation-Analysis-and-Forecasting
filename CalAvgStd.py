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
        # avg_df = avg_df.append(avg_series, ignore_index=True)
        avg_df = pd.concat([avg_df, avg_series.to_frame().T], ignore_index=True)
    # 返回处理后的dataframe
    return avg_df


def deal_txt_data_folds(path, save_path):
    # 创建一个名为"sequences"的文件夹
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 定义一个空的DataFrame来存储所有文件中的数据
    all_data = pd.DataFrame()
    # 遍历文件夹中的所有文件
    for filename in os.listdir(path):
        # 检查文件名是否以.txt结尾
        if filename.endswith('.txt'):
            # 拼接文件路径
            print(f"正在收集{filename}\n")
            file_path = os.path.join(path, filename)
            # 读取文件
            df = pd.read_csv(file_path, sep="\t", header=1, encoding='gb18030')
            data = df.dropna(axis=1)
            features = data.columns.drop(["time", "Diff_Ave_R3", "Tilt_R3", "Diff_Ave_R5", "Tilt_R5"])
            data = average_time_series_with_pandas(data[features], s=1)
            diff = data["Diff_Ave_F3"]
            # 遍历所有可能的起始时间点
            seq_len = 120
            pred_len = 3
            step = 5
            for j in range(0, len(data) - (seq_len + pred_len), step):
                # 获取该时间段内的所有特征值，并转换为numpy数组
                x = data[j:j + seq_len].to_numpy()
                x_frame = data[j:j + seq_len]
                # 获取该时间段结束时刻后m秒的Diff_Ave_F3值，并转换为numpy数组
                y = diff[j + seq_len + pred_len - 1]
                # 新增一个过滤机制来过滤掉等待期间的预测
                if y != x[-1, 2] and y != x[-2, 2] and y != x[-3, 2]:
                    # 将数据追加到all_data中
                    all_data = pd.concat([all_data, x_frame], ignore_index=True)

    # 计算all_data中每一列的平均数和标准差，并保存为一个新的DataFrame
    stats = all_data.agg(["mean", "std", "min", "max"])
    # 打印结果
    print(stats)
    # 保存为txt文件，假设文件名为stats.txt，保存在当前目录下
    stats.to_csv(save_path + "stats.txt", sep="\t", encoding='gb18030')


if __name__ == '__main__':
    deal_txt_data_folds(path="E:\\LSTM数据集\\带跑偏量的_45个属性\\230710\\", save_path="D:\\Desktop\\days3_features41_step5_filtration\\")

import os
import pandas as pd
import numpy as np


mean = -0.270481045308382
std = 5.229984508
root_dir = f'D:\\Desktop\\days3_features41_step5_filtration\\Test\\'
index = 73
file_list = [file for file in os.listdir(root_dir) if file.endswith('.pkl')]
file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
file_path = f'D:\\Desktop\\days3_features41_step5_filtration\\Test\\{file_list[index]}'
data = pd.read_pickle(file_path)
a = np.array(data[0][:, 2]) * std + mean
np.savetxt(f"D:\\Desktop\\MakeGIF\\paper\\attention_attention_shuffle_40\\samples\\sample{index}.txt", a)
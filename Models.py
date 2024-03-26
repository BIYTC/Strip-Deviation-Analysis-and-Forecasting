import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(120, d_model)
        position = torch.arange(0, 120).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 不需要转置
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return self.dropout(x)
        # return x


# 定义一个多头自注意力层带分组打乱
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, groups=5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # shuffle分组数
        self.groups = groups

        # 为每个头定义全连接层
        self.query_layers = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(num_heads)])

        self.out = nn.Linear(d_model, d_model)

    def forward(self, src, k, v, mask=None):
        # 获取批次大小和序列长度
        batch_size, seq_len, d = src.size()
        # 打乱通道顺序
        assert d % self.groups == 0
        channels_per_group = d // self.groups
        # split into groups
        src = src.view(batch_size, seq_len, self.groups, channels_per_group)
        # transpose 1, 2 axis
        src = src.transpose(-1, -2).contiguous()
        # reshape into orignal
        src = src.view(batch_size, seq_len, d)


        # 分割输入数据为多个头
        q, k, v = [self.split_heads(x, self.num_heads, self.d_k) for x in (src, src, src)]

        # 为每个头应用全连接层
        qs, ks, vs = [], [], []
        for i in range(self.num_heads):
            qs.append(self.query_layers[i](q[:, i, :, :]))
            ks.append(self.key_layers[i](k[:, i, :, :]))
            vs.append(self.value_layers[i](v[:, i, :, :]))

        # 应用注意力机制
        heads = []
        for i in range(self.num_heads):
            scores = torch.matmul(qs[i], ks[i].transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
            heads.append(torch.matmul(p_attn, vs[i]))

        # 合并头部
        concat = torch.cat(heads, dim=-1)

        # 应用最终线性层
        return self.out(concat)

    def split_heads(self, x, num_heads, d_k):
        """分割输入数据为多个头"""
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, num_heads, d_k)
        return x.transpose(1, 2)



# 定义一个Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 定义一个多头自注意力层
        self.self_attn = MultiHeadAttention(d_model=d_model,
                                            num_heads=num_heads)

        # 定义一个前馈神经网络层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        # 定义一个层归一化层
        self.layer_norm = nn.LayerNorm(d_model)

        # 定义一个残差连接层
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):
        # 对输入进行自注意力操作，并且获取注意力权重
        src2 = self.self_attn(src, src, src, mask=src_mask)

        # 对自注意力的输出进行残差连接和层归一化
        src = src + self.dropout(src2)
        src = self.layer_norm(src)

        # 对自注意力的输出进行前馈神经网络操作
        src2 = self.feed_forward(src)

        # 对前馈神经网络的输出进行残差连接和层归一化
        src = src + self.dropout(src2)
        src = self.layer_norm(src)

        # 返回输出和注意力权重
        return src


# 定义一个Transformer编码器模型
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(d_model)
        # 定义一个列表，存储多个Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model,
                                    num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        # src = self.pos_encoder(src)
        # 对每一层进行循环，对输入进行编码，并且保存注意力权重
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)

        # 返回输出和注意力权重列表
        return src


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (1, batch_size, hidden_size)
        # c_n: (1, batch_size, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


# 定义注意力机制
class FeatureAttention(nn.Module):
    def __init__(self, hidden_size):
        super(FeatureAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, q, k, v):
        # q: (batch_size, hidden_size)
        # k: (batch_size, hidden_size)
        # v: (batch_size,)

        # 计算注意力权重
        # attn_weights: (batch_size,)
        # 效仿transformer在softmax前做除法
        attn_weights = torch.sum(q * k, dim=-1) / math.sqrt(self.hidden_size)
        attn_output = attn_weights * v
        return attn_output


# 定义时序预测模型
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(TimeSeriesPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        # 位置编码加在这里
        self.pos_encoder = PositionalEncoding(2*input_size)
        # 定义三个LSTM模型，分别命名为LSTMq，LSTMk和LSTMv
        self.LSTMq = LSTM(1, hidden_size)
        self.LSTMk = LSTM(1, hidden_size)
        self.LSTMv = LSTM(1, 1)

        # 定义一个注意力机制，用于计算不同变量或特征之间的注意力权重
        self.attention = FeatureAttention(hidden_size)
        # 定义一个注意力机制，用于计算不同时间点之间的注意力权重
        self.my_transformer = TransformerEncoder(num_layers=2, d_model=2*input_size, num_heads=5)
        # 主体预测部分
        # 创建LSTM网络
        self.lstm = nn.LSTM(input_size * 3, hidden_size, 3, batch_first=True, dropout=0.1, bidirectional=True)
        # 创建全连接层
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # 将数据转移到GPU上
        x = x.to(self.device)
        # x = self.pos_encoder(x)
        # 对输入数据进行拆分，将第三维分解为input_szie个子张量，每个子张量都是一个(batch_szie, seq_len)形状的张量
        x_list = torch.split(x, 1, dim=-1)

        # 对每个子张量分别输入到LSTMq，LSTMk和LSTMv中，得到input_szie个输出张量
        q_list = []
        k_list = []
        v_list = []

        for x_i in x_list:
            output_i, (h_n_i, c_n_i) = self.LSTMq(x_i)
            q_list.append(h_n_i.squeeze(0))
            output_i, (h_n_i, c_n_i) = self.LSTMk(x_i)
            k_list.append(h_n_i.squeeze(0))
            output_i, (h_n_i, c_n_i) = self.LSTMv(x_i)
            v_list.append(h_n_i.squeeze(0).squeeze(-1))
        attn_output_list = []
        b_list = []
        for i in range(self.input_size):
            q_i = q_list[i]
            x_i = x_list[i].squeeze(-1)
            b_i = []
            attn_output_i = torch.zeros(x.size(0), x.size(1)).to(self.device)  # 注意将结果也转移到GPU上
            for j in range(self.input_size):
                k_j = k_list[j]
                v_j = v_list[j]
                b_ij = self.attention(q_i, k_j, v_j)
                b_i.append(b_ij)
            b_i_torch = torch.stack(b_i, -1)
            softmax_b_i = torch.softmax(b_i_torch, dim=-1)
            b_list.append(softmax_b_i)
            attn_output_i = torch.squeeze(torch.bmm(x, torch.unsqueeze(softmax_b_i, -1)), dim=-1)
            attn_output_list.append(attn_output_i)
        # 将得到的加权数据重新组合成一个(batch_size, seq_len, input_size)形状的张量
        attn_output = torch.stack(attn_output_list, dim=-1)
        b = torch.stack(b_list, dim=-1)
        # 将通道拼接
        attn_output = torch.cat((x, attn_output), dim=-1)
        attn_output = self.pos_encoder(attn_output)

        # 后面继续接时间注意力机制
        out = self.my_transformer(attn_output)

        # 再进行一次拼接，将原始数据拼接进去
        out = torch.cat((x, out), dim=-1)
        # 通过LSTM网络得到输出和最终的隐藏状态和细胞状态
        out, (hn, cn) = self.lstm(out)
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        out = F.relu(out)
        # 通过全连接层得到最终的预测值
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

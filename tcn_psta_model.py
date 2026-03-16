import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.psta import PSTA

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 输入shape: [N, C, T, V, M]
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=(kernel_size, 1),
                              stride=(stride, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=(kernel_size, 1),
                              stride=(stride, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                                self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 输入x shape: [N, C, T, V, M]
        N, C, T, V, M = x.size()
        
        # 处理多人的情况，转换为[N*M, C, T, V]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [N, M, C, T, V]
        x = x.view(N*M, C, T, V)  # [N*M, C, T, V]
        
        # 保存原始输入用于残差连接
        identity = x
        
        # 通过网络
        out = self.net(x)  # [N*M, C, T, V]
        
        # 处理残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # 确保维度匹配
        if out.size(2) != identity.size(2):
            # 计算需要裁剪的大小
            diff = out.size(2) - identity.size(2)
            if diff > 0:
                # 裁剪out
                out = out[:, :, :-diff, :]
            else:
                # 裁剪identity
                identity = identity[:, :, :-(-diff), :]
        
        # 添加残差连接
        out = self.relu(out + identity)  # [N*M, C, T, V]
        
        # 恢复原始shape [N, C, T, V, M]
        out = out.view(N, M, -1, T, V)  # [N, M, C, T, V]
        out = out.permute(0, 2, 3, 4, 1).contiguous()  # [N, C, T, V, M]
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 输入x shape: [N, C, T, V, M]
        return self.network(x)  # 输出shape: [N, C, T, V, M]

class TCNPSTAModel(nn.Module):
    def __init__(self, num_class=2, num_point=18, num_person=1, in_channels=3,
                 dropout=0.5, pretrained_weights=None):
        super(TCNPSTAModel, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        
        # 数据标准化
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # TCN特征提取器
        self.tcn = TemporalConvNet(
            num_inputs=in_channels,
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=dropout
        )
        
        # PSTA模块
        self.psta = PSTA(256)
        
        # 全局特征聚合
        self.fcn = nn.Conv2d(256, 256, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_class)
        
        self._init_weights()
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, pretrained_weights):
        print(f"加载预训练权重: {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            pretrained_dict[k] = v
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 层")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        # 输入x shape: [N, C, T, V, M]
        N, C, T, V, M = x.size()
        
        # 数据标准化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N, C, T, V, M)
        
        # TCN特征提取
        x = self.tcn(x)  # [N, 256, T, V, M]
        
        # PSTA模块
        x = self.psta(x)  # [N, 256, T, V, M]
        
        # 全局特征聚合
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
        x = self.fcn(x)
        x = self.gap(x).view(N*M, -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(N, M, -1).mean(dim=1)
        return x

def create_tcn_psta_model(num_class=2, pretrained_path=None):
    model = TCNPSTAModel(
        num_class=num_class,
        num_point=18,
        num_person=1,
        in_channels=3,
        dropout=0.5,
        pretrained_weights=pretrained_path
    )
    return model 
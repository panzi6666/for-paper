import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ----------- 基础图卷积模块 -----------
class Graph():
    def __init__(self, layout='openpose', strategy='spatial'):
        self.num_node = 18
        self.edge = self._get_edge(layout)
        self.A = self._get_adjacency(strategy)
    def _get_edge(self, layout):
        if layout == 'openpose':
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                            (0, 14), (14, 15), (10, 17), (13, 16)]
            self.center = 1
            return self_link, neighbor_link
        else:
            raise ValueError(f"不支持的布局: {layout}")
    def _get_adjacency(self, strategy):
        A = np.zeros((self.num_node, self.num_node))
        self_link, neighbor_link = self.edge
        for i, j in self_link:
            A[j, i] = 1
        for i, j in neighbor_link:
            A[j, i] = 1
            A[i, j] = 1
        if strategy == 'spatial':
            return [A]
        elif strategy == 'st_cbam':
            A1 = A.copy()
            A2, A3 = np.zeros_like(A), np.zeros_like(A)
            A2 = np.minimum(np.matmul(A, A), 1) - A
            A3 = np.minimum(np.matmul(A2, A), 1) - A - A2
            hip_knee_ankle_links = [(8, 9), (9, 10), (11, 12), (12, 13)]
            for i, j in hip_knee_ankle_links:
                A1[j, i] *= 2
                A1[i, j] *= 2
            return [A1, A2, A3]
        else:
            raise ValueError(f"不支持的策略: {strategy}")

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True):
        if isinstance(A, list):
            A = np.array(A)
        super(GCN, self).__init__()
        self.num_subset = A.shape[0]
        self.out_channels = out_channels
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = torch.from_numpy(A.astype(np.float32))
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subset, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        N, C, T, V = x.size()
        y = self.conv(x)
        y = y.view(N, self.num_subset, self.out_channels, T, V)
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.to(x.device)
        z = torch.zeros(N, self.out_channels, T, V).to(y.device)
        for i, a in enumerate(A):
            z = z + torch.einsum('nctv,vw->nctw', (y[:, i], a))
        z = self.bn(z)
        z = self.relu(z)
        return z

# ----------- 通道注意力机制（CBAM） -----------
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, t, v, m = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * out

# ----------- ST-CBAM Block -----------
class ST_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        if isinstance(A, list):
            A = np.array(A)
        super(ST_CBAM_Block, self).__init__()
        self.gcn = GCN(in_channels, out_channels, A, adaptive=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = None
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Identity()
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
    def forward(self, x):
        N, C, T, V, M = x.size()
        x_ = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
        res = self.residual(x_)
        if self.residual_conv is not None:
            res = self.residual_conv(x_)
        x_ = self.gcn(x_)
        x_ = x_ + res
        x_ = self.relu(x_)
        x_ = x_.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1)
        x_ = self.channel_attention(x_)
        return x_

# ----------- 主模型 -----------
class STCBAM_NoPSTA_Net(nn.Module):
    def __init__(self, num_class=2, num_point=18, num_person=1, in_channels=3,
                 dropout=0.5, pretrained_weights=None):
        super(STCBAM_NoPSTA_Net, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        graph = Graph(layout='openpose', strategy='st_cbam')
        A = graph.A
        if isinstance(A, list):
            A = np.array(A)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.stcbam_layers = nn.ModuleList((
            ST_CBAM_Block(in_channels, 64, A, residual=False),
            ST_CBAM_Block(64, 64, A),
            ST_CBAM_Block(64, 64, A),
            ST_CBAM_Block(64, 128, A),
            ST_CBAM_Block(128, 128, A),
            ST_CBAM_Block(128, 256, A),
            ST_CBAM_Block(256, 256, A),
        ))
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N, M * C, T, V)
        x = x.view(N, C, T, V, M)
        for layer in self.stcbam_layers:
            x = layer(x)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
        x = self.fcn(x)
        x = self.gap(x).view(N*M, -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(N, M, -1).mean(dim=1)
        return x

def create_stcbam_nopsta_model(num_class=2, pretrained_path=None):
    model = STCBAM_NoPSTA_Net(
        num_class=num_class,
        num_point=18,
        num_person=1,
        in_channels=3,
        dropout=0.5,
        pretrained_weights=pretrained_path
    )
    return model 
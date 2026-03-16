import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ---------------------  ST-GCN 基础组件  ---------------------

class Graph():
    """
    图结构类，定义关节点之间的连接关系
    """
    def __init__(self, layout='openpose', strategy='spatial'):
        self.num_node = 18  # OpenPose 18关键点
        self.edge = self._get_edge(layout)
        self.A = self._get_adjacency(strategy)
        
    def _get_edge(self, layout):
        if layout == 'openpose':
            # OpenPose 18关键点的标准连接关系
            # 0-头顶, 1-颈部, 2-右肩, 3-右肘, 4-右腕, 5-左肩, 6-左肘, 7-左腕,
            # 8-右髋, 9-右膝, 10-右踝, 11-左髋, 12-左膝, 13-左踝, 14-眼睛中心, 15-鼻子, 16-左脚, 17-右脚
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                            (0, 14), (14, 15), (10, 17), (13, 16)]
            self.center = 1  # 颈部作为中心点
            return self_link, neighbor_link
        else:
            raise ValueError(f"不支持的布局: {layout}")
    
    def _get_adjacency(self, strategy):
        A = np.zeros((self.num_node, self.num_node))
        self_link, neighbor_link = self.edge
        
        # 添加自连接
        for i, j in self_link:
            A[j, i] = 1
        
        # 添加邻居连接
        for i, j in neighbor_link:
            A[j, i] = 1
            A[i, j] = 1  # 双向连接
            
        if strategy == 'spatial':
            # 基本的空间连接
            return [A]
        elif strategy == 'st_cbam':
            # 针对ST-CBAM的改进连接
            # 包括基本连接、可达连接和强化关节关系
            A1 = A.copy()  # 1-hop连接
            A2, A3 = np.zeros_like(A), np.zeros_like(A)
            
            # 2-hop连接 (朋友的朋友)
            A2 = np.minimum(np.matmul(A, A), 1) - A
            
            # 3-hop连接 
            A3 = np.minimum(np.matmul(A2, A), 1) - A - A2
            
            # 强化重要关节连接 (针对深蹲动作)
            # 增强髋-膝-踝连接的权重
            hip_knee_ankle_links = [(8, 9), (9, 10), (11, 12), (12, 13)]
            for i, j in hip_knee_ankle_links:
                A1[j, i] *= 2  # 加强这些连接的权重
                A1[i, j] *= 2
            
            return [A1, A2, A3]
        else:
            raise ValueError(f"不支持的策略: {strategy}")


class GCN(nn.Module):
    """
    图卷积网络基础模块
    """
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True):
        if isinstance(A, list):
            A = np.array(A)
        super(GCN, self).__init__()
        self.num_subset = A.shape[0]
        self.out_channels = out_channels
        self.adaptive = adaptive
        
        # 可学习的邻接矩阵
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = torch.from_numpy(A.astype(np.float32))
            
        # 图卷积层
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subset, kernel_size=1)
        
        # 批标准化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        N, C, T, V = x.size()
        y = self.conv(x)  # [N, out_channels * num_subset, T, V]
        y = y.view(N, self.num_subset, self.out_channels, T, V)  # [N, num_subset, out_channels, T, V]
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.to(x.device)
        z = torch.zeros(N, self.out_channels, T, V).to(y.device)  # [N, out_channels, T, V]
        for i, a in enumerate(A):
            z = z + torch.einsum('nctv,vw->nctw', (y[:, i], a))
        z = self.bn(z)
        z = self.relu(z)
        return z


# ---------------------  通道注意力机制 (SE模块)  ---------------------

class ChannelAttention(nn.Module):
    """
    基于SE (Squeeze-and-Excitation) 的通道注意力机制
    为关节点特征提供动态权重
    """
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入 x: [N, C, T, V, M]
        b, c, t, v, m = x.size()
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 共享MLP
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        # 合并并应用sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        
        # 返回缩放后的输入
        return x * out


# ---------------------  ST-CBAM模块 (结合GCN和通道注意力)  ---------------------

class ST_CBAM_Block(nn.Module):
    """
    时空通道注意力联合建模块 (ST-CBAM)
    结合GCN和通道注意力机制
    输入输出均为[N, C, T, V, M]
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        if isinstance(A, list):
            A = np.array(A)
        super(ST_CBAM_Block, self).__init__()
        self.gcn = GCN(in_channels, out_channels, A, adaptive=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = None
        # 残差连接自动适配shape
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
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.size()
        x_ = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)  # [N*M, C, T, V]
        res = self.residual(x_)
        if self.residual_conv is not None:
            res = self.residual_conv(x_)
        x_ = self.gcn(x_)  # [N*M, out_channels, T, V]
        x_ = x_ + res
        x_ = self.relu(x_)
        x_ = x_.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1)  # [N, out_channels, T, V, M]
        x_ = self.channel_attention(x_)
        return x_


# ---------------------  PSTA模块 (时空注意力并行化融合)  ---------------------

class PositionalEncoding(nn.Module):
    """
    时序位置编码，用于标记动作相位
    """
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦/余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 输入 x: [N, C, T, V, M]
        b, c, t, v, m = x.size()
        
        # 重塑为[b*v*m, c, t]以便对每个关节和每人应用位置编码
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(b*v*m, c, t)
        
        # 确保位置编码的维度与输入匹配
        pe = self.pe[:, :t, :c].permute(0, 2, 1)  # [1, c, t]
        pe = pe.expand(b*v*m, -1, -1)  # [b*v*m, c, t]
        
        # 添加位置编码
        x = x + pe
        
        # 恢复原始形状
        x = x.view(b, v, m, c, t).permute(0, 3, 4, 1, 2)
        
        return x


class CausalSelfAttention(nn.Module):
    """
    具有因果掩码的多头自注意力
    确保模型只能看到过去和当前的信息，而不能看到未来的信息
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
    def forward(self, x):
        # 输入 x: [N, C, T, V, M]
        b, c, t, v, m = x.size()
        
        # 重塑为时间序列形式 [T, N*V*M, C]
        x = x.permute(2, 0, 3, 4, 1).contiguous().view(t, b*v*m, c)
        
        # 创建因果掩码
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(x.device)
        
        # 应用自注意力机制(带有因果掩码)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        
        # 恢复原始形状
        attn_output = attn_output.view(t, b, v, m, c).permute(1, 4, 0, 2, 3)
        
        return attn_output


class GatedFusion(nn.Module):
    """
    门控融合模块
    动态平衡ST-CBAM和TPA (时序位置注意力) 的贡献
    """
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(dim*2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        # 输入 x1, x2: [N, C, T, V, M]
        # x1: ST-CBAM输出，捕获关节稳定性
        # x2: TPA输出，捕获全局动作退化特征
        
        # 连接特征
        cat = torch.cat([x1, x2], dim=1)  # [N, 2C, T, V, M]
        
        # 学习门控权重
        gate = self.gate(cat)  # [N, C, T, V, M]
        
        # 应用门控融合
        return gate * x1 + (1 - gate) * x2


class PSTA_Module(nn.Module):
    """
    时空注意力并行化融合模块 (PSTA)
    输入输出均为[N, C, T, V, M]
    """
    def __init__(self, in_channels, out_channels, A, dropout=0.1):
        if isinstance(A, list):
            A = np.array(A)
        super(PSTA_Module, self).__init__()
        self.st_cbam = ST_CBAM_Block(in_channels, out_channels, A)
        self.position_encoding = PositionalEncoding(in_channels)
        self.causal_self_attention = CausalSelfAttention(in_channels, nhead=8, dropout=dropout)
        self.tpa_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gated_fusion = GatedFusion(out_channels)
        self.proj = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.size()
        stcbam_output = self.st_cbam(x)  # [N, out_channels, T, V, M]
        tpa_input = self.position_encoding(x)  # [N, in_channels, T, V, M]
        tpa_output = self.causal_self_attention(tpa_input)  # [N, in_channels, T, V, M]
        # Conv3d输入5维，M合并到batch
        tpa_output = tpa_output.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V, 1)
        tpa_output = self.tpa_proj(tpa_output)  # [N*M, out_channels, T, V, 1]
        tpa_output = tpa_output.view(N, M, -1, T, V, 1).permute(0, 2, 3, 4, 1, 5).squeeze(-1)  # [N, out_channels, T, V, M]
        # 融合
        fused = self.gated_fusion(stcbam_output, tpa_output)  # [N, out_channels, T, V, M]
        # proj同理
        fused_proj = fused.permute(0, 4, 1, 2, 3).contiguous().view(N*M, fused.size(1), fused.size(2), fused.size(3), 1)
        output = self.proj(fused_proj)  # [N*M, out_channels, T, V, 1]
        output = output.view(N, M, -1, T, V, 1).permute(0, 2, 3, 4, 1, 5).squeeze(-1)  # [N, out_channels, T, V, M]
        # 残差
        if output.shape == x.shape:
            output = output + x
        # LayerNorm和Dropout
        output = output.permute(0, 2, 3, 4, 1).contiguous()  # [N, T, V, M, C]
        output = self.norm(output)
        output = self.dropout(output)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [N, C, T, V, M]
        return output


# ---------------------  主模型数据流重构  ---------------------

class STCBAM_PSTA_Net(nn.Module):
    """
    基于ST-CBAM和PSTA模块的深蹲力竭预测模型
    输入输出均为[N, C, T, V, M]
    """
    def __init__(self, num_class=2, num_point=18, num_person=1, in_channels=3,
                 dropout=0.5, pretrained_weights=None):
        super(STCBAM_PSTA_Net, self).__init__()
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
            PSTA_Module(64, 128, A),
            PSTA_Module(128, 128, A),
            PSTA_Module(128, 256, A),
            PSTA_Module(256, 256, A),
        ))
        self.fcn = nn.Conv2d(256, 256, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_class)
        self._init_weights()
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
    def forward(self, x):
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.size()
        # 数据标准化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N, M * C, T, V)
        x = x.view(N, C, T, V, M)
        # 主干网络
        for layer in self.stcbam_layers:
            x = layer(x)
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
    def _init_weights(self):
        # 初始化网络权重
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


# 函数用于创建完整模型
def create_stcbam_psta_model(num_class=2, pretrained_path=None):
    """
    创建完整的STCBAM-PSTA深蹲力竭预测模型
    
    参数:
        num_class: 分类数量 (2: 力竭/非力竭)
        pretrained_path: 预训练权重路径
    """
    # 保证A为numpy数组（虽然A在Net内部生成，但为保险起见）
    model = STCBAM_PSTA_Net(
        num_class=num_class,
        num_point=18,
        num_person=1,
        in_channels=3,
        dropout=0.5,
        pretrained_weights=pretrained_path
    )
    
    return model


if __name__ == "__main__":
    # 模型测试代码
    model = create_stcbam_psta_model(
        num_class=2, 
        pretrained_path="mmskeleton-master/mmskeleton-master/checkpoints/st_gcn.kinetics-6fa43f73.pth"
    )
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    dummy_input = torch.randn(4, 3, 150, 18, 1)  # [N, C, T, V, M]
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}, 输出形状: {output.shape}") 
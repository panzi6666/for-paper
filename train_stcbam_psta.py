import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from st_cbam_model import create_stcbam_psta_model

# 设置随机种子以确保可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定义数据集类
class SquatDataset(Dataset):
    def __init__(self, data_path, is_train=True, split_ratio=0.8, transform=None):
        # 加载数据
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:  # .npy
            data = np.load(data_path, allow_pickle=True).item()
        
        self.X = data['x']  # 骨架数据 (N, C, T, V, M)
        self.y = data['y']  # 标签 (N,)
        
        # 划分训练/测试集
        n_samples = len(self.y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        split = int(np.floor(split_ratio * n_samples))
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        # 根据is_train选择相应的数据
        if is_train:
            self.X = self.X[train_indices]
            self.y = self.y[train_indices]
        else:
            self.X = self.X[test_indices]
            self.y = self.y[test_indices]
            
        # 数据转换
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # [3, 150, 18, 1]
        y = self.y[idx]
        assert x.shape == (3, 150, 18, 1), f"数据shape异常: {x.shape}"
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

# 数据增强
class RandomSpatialFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        # 随机左右翻转
        if np.random.random() < self.p:
            # 交换左右侧关节点
            # 2-右肩 <-> 5-左肩, 3-右肘 <-> 6-左肘, 4-右腕 <-> 7-左腕
            # 8-右髋 <-> 11-左髋, 9-右膝 <-> 12-左膝, 10-右踝 <-> 13-左踝
            pairs = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 16)]
            for i, j in pairs:
                x[:, :, [i, j]] = x[:, :, [j, i]]
        return x

class RandomTemporalCrop:
    def __init__(self, crop_ratio=0.9):
        self.crop_ratio = crop_ratio
        
    def __call__(self, x):
        # 随机裁剪时间维度
        c, t, v, m = x.shape
        new_t = int(t * self.crop_ratio)
        start = np.random.randint(0, t - new_t + 1)
        return x[:, start:start+new_t, :, :]

class RandomNoise:
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        
    def __call__(self, x):
        # 添加高斯噪声
        noise = np.random.normal(0, self.sigma, x.shape)
        return x + noise

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc='训练')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算性能指标
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': train_loss / (pbar.n + 1),
            'acc': 100. * train_correct / train_total,
        })
    
    return train_loss / len(train_loader), 100. * train_correct / train_total

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='验证'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 计算性能指标
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            # 收集预测和目标用于计算更多指标
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 计算F1分数和其他指标
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    
    print('\n验证集性能指标:')
    print(classification_report(all_targets, all_preds))
    
    return (val_loss / len(val_loader), 
            100. * val_correct / val_total, 
            f1, precision, recall)

# 绘制训练曲线
def plot_learning_curves(train_loss, train_acc, val_loss, val_acc, save_dir):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.legend()
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='训练准确率')
    plt.plot(val_acc, label='验证准确率')
    plt.legend()
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

# 主函数
def main():
    parser = argparse.ArgumentParser(description='训练ST-CBAM-PSTA深蹲力竭预测模型')
    parser.add_argument('--data_path', type=str, default='models/squat_dataset.pkl', 
                        help='处理后的数据集路径')
    parser.add_argument('--pretrained', type=str,
                        default='mmskeleton-master/mmskeleton-master/checkpoints/st_gcn.kinetics-6fa43f73.pth',
                        help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--save_dir', type=str, default='models/stcbam_psta_results', 
                        help='保存结果的目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'使用设备: {device}')
    
    # 准备数据集
    train_transform = None
    train_dataset = SquatDataset(args.data_path, is_train=True, transform=train_transform)
    val_dataset = SquatDataset(args.data_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    # 创建模型
    model = create_stcbam_psta_model(
        num_class=2,  # 二分类: 力竭/非力竭
        pretrained_path=args.pretrained
    )
    model = model.to(device)
    print(f'模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    
    # 训练循环
    best_f1 = 0
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练和验证
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(model, val_loader, criterion, device)
        
        # 学习率调整
        scheduler.step(val_f1)
        
        # 保存历史记录
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_precision': val_precision, 
                'val_recall': val_recall
            }, model_path)
            print(f'模型已保存到 {model_path}')
            
        # 绘制学习曲线
        plot_learning_curves(
            train_loss_history, 
            train_acc_history, 
            val_loss_history, 
            val_acc_history, 
            args.save_dir
        )
        
        # 输出当前结果
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, '
              f'Val Recall: {val_recall:.4f}')
    
    print(f'训练完成！最佳验证F1: {best_f1:.4f}')

if __name__ == '__main__':
    main() 
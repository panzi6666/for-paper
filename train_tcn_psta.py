import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.data import SquatDataset
from tcn_psta_model import create_tcn_psta_model
from utils.utils import set_seed, train, validate, plot_learning_curves
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SquatDataset(Dataset):
    def __init__(self, data_path, is_train=True, split_ratio=0.8, transform=None):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据格式正确
        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                self.data = data['x']
                self.labels = data['y']
            else:
                raise ValueError("数据字典必须包含'x'和'y'键")
        else:
            raise ValueError("数据必须是字典格式")
        
        # 划分训练集和验证集
        n_samples = len(self.data)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * split_ratio)
        
        if is_train:
            self.data = self.data[indices[:split_idx]]
            self.labels = self.labels[indices[:split_idx]]
        else:
            self.data = self.data[indices[split_idx:]]
            self.labels = self.labels[indices[split_idx:]]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return torch.FloatTensor(data), torch.LongTensor([label])[0]

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算评估指标
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    accuracy = sum(1 for x, y in zip(all_preds, all_targets) if x == y) / len(all_targets)
    
    return total_loss / len(val_loader), accuracy, f1, precision, recall

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset pickle file')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据加载器
    train_dataset = SquatDataset(args.data_path, is_train=True)
    val_dataset = SquatDataset(args.data_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_tcn_psta_model(num_class=2, pretrained_path=args.pretrained).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, f1, precision, recall = validate(model, val_loader, criterion, device)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs,
                        os.path.join(args.save_dir, 'learning_curves.png'))
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    with open(os.path.join(args.save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main() 
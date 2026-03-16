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
from stgcn_nopsta_model import create_stcbam_nopsta_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SquatDataset(Dataset):
    def __init__(self, data_path, is_train=True, split_ratio=0.8, transform=None):
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = np.load(data_path, allow_pickle=True).item()
        self.X = data['x']
        self.y = data['y']
        n_samples = len(self.y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split = int(np.floor(split_ratio * n_samples))
        train_indices = indices[:split]
        test_indices = indices[split:]
        if is_train:
            self.X = self.X[train_indices]
            self.y = self.y[train_indices]
        else:
            self.X = self.X[test_indices]
            self.y = self.y[test_indices]
        self.transform = transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        assert x.shape == (3, 150, 18, 1), f"数据shape异常: {x.shape}"
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    pbar = tqdm(train_loader, desc='训练')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({
            'loss': train_loss / (pbar.n + 1),
            'acc': 100. * train_correct / train_total,
        })
    return train_loss / len(train_loader), 100. * train_correct / train_total

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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    print('\n验证集性能指标:')
    print(classification_report(all_targets, all_preds))
    return (val_loss / len(val_loader), 
            100. * val_correct / val_total, 
            f1, precision, recall)

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

def main():
    parser = argparse.ArgumentParser(description='训练ST-CBAM（无PSTA）深蹲力竭预测模型')
    parser.add_argument('--data_path', type=str, default='models/squat_dataset.pkl', 
                        help='处理后的数据集路径')
    parser.add_argument('--pretrained', type=str,
                        default='mmskeleton-master/mmskeleton-master/checkpoints/st_gcn.kinetics-6fa43f73.pth',
                        help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--save_dir', type=str, default='models/stgcn_nopsta_results', 
                        help='保存结果的目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'使用设备: {device}')
    train_transform = None
    train_dataset = SquatDataset(args.data_path, is_train=True, transform=train_transform)
    val_dataset = SquatDataset(args.data_path, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    model = create_stcbam_nopsta_model(
        num_class=2,
        pretrained_path=args.pretrained
    )
    model = model.to(device)
    print(f'模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    best_f1 = 0
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
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
        plot_learning_curves(
            train_loss_history, 
            train_acc_history, 
            val_loss_history, 
            val_acc_history, 
            args.save_dir
        )
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, '
              f'Val Recall: {val_recall:.4f}')
    print(f'训练完成！最佳验证F1: {best_f1:.4f}')

if __name__ == '__main__':
    main() 
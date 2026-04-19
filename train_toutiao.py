import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import swanlab
import numpy as np


# ========== 1. 定义 Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ========== 2. FGM 对抗训练类 ==========
class FGM:
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embedding' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ========== 3. 数据加载与预处理 ==========
def load_toutiao_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                data.append({'text': parts[3], 'label': parts[2]})
    return pd.DataFrame(data)


train_df = load_toutiao_data('train_3k.txt')
dev_df = load_toutiao_data('dev_1k.txt')
test_df = load_toutiao_data('test_1k.txt')

unique_labels = sorted(set(train_df['label'].unique()))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(unique_labels)

train_df['label_id'] = train_df['label'].map(label2id)
dev_df['label_id'] = dev_df['label'].map(label2id)
test_df['label_id'] = test_df['label'].map(label2id)

# 计算类别权重用于Focal Loss
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label_id']), y=train_df['label_id'])
alpha_tensor = torch.tensor(class_weights, dtype=torch.float)

# ========== 4. 模型加载 ==========
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# ========== 5. 数据集封装 ==========
class ToutiaoDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}


# 超参数配置
CONFIG = {
    "learning_rate": 3e-5,
    "batch_size": 16,
    "epochs": 3,
    "max_length": 128,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fgm_epsilon": 0.5,  # FGM扰动幅度
    "focal_gamma": 2.0,  # Focal Loss聚焦参数
}

train_dataset = ToutiaoDataset(train_df['text'].values, train_df['label_id'].values, tokenizer, CONFIG['max_length'])
dev_dataset = ToutiaoDataset(dev_df['text'].values, dev_df['label_id'].values, tokenizer, CONFIG['max_length'])
test_dataset = ToutiaoDataset(test_df['text'].values, test_df['label_id'].values, tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=CONFIG['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}, Config: {CONFIG}")

# 优化器
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
total_steps = len(train_loader) * CONFIG['epochs']
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(CONFIG['warmup_ratio'] * total_steps),
                             num_training_steps=total_steps)

# 损失函数（使用Focal Loss）

criterion = FocalLoss(alpha=alpha_tensor.to(device), gamma=CONFIG['focal_gamma'])

# 初始化FGM
fgm = FGM(model, epsilon=CONFIG['fgm_epsilon'])

# 初始化SwanLab（云端+本地）
swanlab.init(
    project="toutiao-text-classification",
    experiment_name="bert-focal-fgm-bs16-lr3e-5",  # 实验名称
    config=CONFIG,
    mode="cloud"  # 可选项: "cloud"（默认，需要登录）, "local"（仅本地）, "disabled"（禁用）
)


# ========== 6. 评估函数（包含可视化数据收集） ==========
def evaluate(loader, model, device, criterion, stage="val"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    for batch in tqdm(loader, desc=f"Evaluating {stage}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    # 记录指标到 SwanLab
    swanlab.log({
        f"{stage}/loss": avg_loss,
        f"{stage}/accuracy": acc,
        f"{stage}/precision": precision,
        f"{stage}/recall": recall,
        f"{stage}/f1": f1,
    })

    return avg_loss, acc, precision, recall, f1, all_labels, all_preds


# ========== 7. 训练循环 ==========
print("开始训练（集成Focal Loss + FGM对抗训练）...")
best_val_acc = 0

for epoch in range(CONFIG['epochs']):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 第一步：正常前向传播和反向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        total_loss += loss.item()

        # 第二步：FGM 对抗训练
        fgm.attack()  # 添加扰动
        outputs_adv = model(input_ids, attention_mask=attention_mask)
        loss_adv = criterion(outputs_adv.logits, labels)
        loss_adv.backward()  # 计算对抗样本的梯度
        fgm.restore()  # 恢复embedding参数

        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item(), 'loss_adv': loss_adv.item()})

    avg_train_loss = total_loss / len(train_loader)
    swanlab.log({"train/loss": avg_train_loss, "train/learning_rate": lr_scheduler.get_last_lr()[0]})

    # 验证
    val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(dev_loader, model, device, criterion, stage="val")
    print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with Val Acc: {best_val_acc:.4f}")

# ========== 8. 最终测试及详细报告 ==========
print("\n===== 最终测试集评估 =====")
test_loss, test_acc, test_prec, test_rec, test_f1, test_true, test_pred = evaluate(test_loader, model, device,
                                                                                   criterion, stage="test")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test F1: {test_f1:.4f}")

# 打印详细的分类报告
print("\n分类报告（各类别详细指标）:")
report = classification_report(test_true, test_pred, target_names=unique_labels, zero_division=0)
print(report)

# 1. 计算并记录混淆矩阵
# 修正混淆矩阵记录
swanlab.log({
    "test/confusion_matrix": swanlab.confusion_matrix(
        test_true,               # ✅ 正确，按顺序传入真实标签
        test_pred,               # ✅ 正确，按顺序传入预测标签
        unique_labels            # ✅ 正确，按顺序传入类别名称
    )
})


# # 2. 计算各类别F1分数，用于绘制柱状图
# from sklearn.metrics import f1_score
#
# per_class_f1 = f1_score(test_true, test_pred, average=None, zero_division=0)
# class_f1_dict = {unique_labels[i]: per_class_f1[i] for i in range(len(unique_labels))}
# swanlab.log({
#     "test/class_f1": swanlab.Table(
#         columns=["类别", "F1分数"],
#         data=[[cls, score] for cls, score in class_f1_dict.items()]
#     )
# })

print(f"\n最佳模型验证准确率: {best_val_acc:.4f}")
print(f"最终测试准确率: {test_acc:.4f}")
print(f"测试F1分数: {test_f1:.4f}")

swanlab.finish()
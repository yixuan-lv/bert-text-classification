import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import ToutiaoDataset
from model import BertClassifier
from utils import load_toutiao_data, collate_fn, evaluate

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

# 1. 数据加载
train_df = load_toutiao_data(CONFIG['data_path']['train'])
dev_df = load_toutiao_data(CONFIG['data_path']['dev'])
test_df = load_toutiao_data(CONFIG['data_path']['test'])

unique_labels = sorted(set(train_df['label'].unique()))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(unique_labels)

train_df['label_id'] = train_df['label'].map(label2id)
dev_df['label_id'] = dev_df['label'].map(label2id)
test_df['label_id'] = test_df['label'].map(label2id)

# 2. 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = BertClassifier(CONFIG['model_name'], num_labels, dropout=CONFIG.get('dropout', 0.1))

# 3. Dataset & DataLoader
train_dataset = ToutiaoDataset(train_df['text'].values, train_df['label_id'].values)
dev_dataset = ToutiaoDataset(dev_df['text'].values, dev_df['label_id'].values)
test_dataset = ToutiaoDataset(test_df['text'].values, test_df['label_id'].values)

def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, CONFIG['max_length'])
    )

train_loader = create_loader(train_dataset, CONFIG['batch_size'], shuffle=True)
dev_loader = create_loader(dev_dataset, CONFIG['batch_size'])
test_loader = create_loader(test_dataset, CONFIG['batch_size'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}, Config: {CONFIG}")

# 4. 优化器、调度器、损失函数
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
total_steps = len(train_loader) * CONFIG['epochs']
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(CONFIG['warmup_ratio'] * total_steps),
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()

# 5. 训练循环
print("开始训练（自定义BERT模型 + 交叉熵）...")
best_val_acc = 0
for epoch in range(CONFIG['epochs']):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    val_acc, _, _ = evaluate(dev_loader, model, device)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with Val Acc: {best_val_acc:.4f}")

# 6. 最终测试
print("\n===== 加载最佳模型进行最终测试 =====")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
test_acc, test_true, test_pred = evaluate(test_loader, model, device)
print(f"Test Accuracy: {test_acc:.4f}")

print("\n分类报告:")
print(classification_report(test_true, test_pred, target_names=unique_labels, zero_division=0))

# 混淆矩阵
cm = confusion_matrix(test_true, test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("混淆矩阵已保存为 confusion_matrix.png")

# F1柱状图
per_class_f1 = f1_score(test_true, test_pred, average=None, zero_division=0)
sorted_idx = np.argsort(per_class_f1)
sorted_classes = [unique_labels[i] for i in sorted_idx]
sorted_f1 = per_class_f1[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_classes, sorted_f1, color='skyblue')
plt.xlabel('F1 Score')
plt.title('Per-class F1 Scores on Test Set')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('per_class_f1.png', dpi=300)
plt.close()
print("各类别 F1 柱状图已保存为 per_class_f1.png")
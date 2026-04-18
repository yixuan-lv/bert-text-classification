import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import swanlab


# ========== 1. 数据加载与预处理 ==========
def load_toutiao_data(file_path):
    """加载今日头条数据，提取标题和类别"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                data.append({
                    'text': parts[3],  # 标题
                    'label': parts[2]  # 分类名称
                })
    return pd.DataFrame(data)


# 加载数据
train_df = load_toutiao_data('train_3k.txt')
dev_df = load_toutiao_data('dev_1k.txt')
test_df = load_toutiao_data('test_1k.txt')

# 创建标签映射
unique_labels = sorted(set(train_df['label'].unique()))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)
print(f"类别数: {num_labels}")
print(f"训练集大小: {len(train_df)}, 验证集: {len(dev_df)}, 测试集: {len(test_df)}")

# 添加 label_id 列
train_df['label_id'] = train_df['label'].map(label2id)
dev_df['label_id'] = dev_df['label'].map(label2id)
test_df['label_id'] = test_df['label'].map(label2id)

# ========== 2. 加载模型和分词器 ==========
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# ========== 3. 数据集封装 ==========
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
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 创建 Dataset 和 DataLoader
batch_size = 32
train_dataset = ToutiaoDataset(train_df['text'].values, train_df['label_id'].values, tokenizer)
dev_dataset = ToutiaoDataset(dev_df['text'].values, dev_df['label_id'].values, tokenizer)
test_dataset = ToutiaoDataset(test_df['text'].values, test_df['label_id'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ========== 4. 训练准备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")

# 超参数
epochs = 3
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# 初始化 SwanLab
swanlab.init(
    project="toutiao-text-classification",
    experiment_name="bert-cpu-run",
    config={
        "model": "bert-base-chinese",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_length": 128,
        "optimizer": "AdamW",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "device": str(device)
    }
)


# ========== 5. 训练与评估函数 ==========
def evaluate(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="评估中"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return acc, prec, rec, f1


print("开始训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_loader)

    # 验证
    val_acc, val_prec, val_rec, val_f1 = evaluate(dev_loader, model, device)

    # 记录到 SwanLab
    swanlab.log({
        "train/loss": avg_train_loss,
        "val/accuracy": val_acc,
        "val/precision": val_prec,
        "val/recall": val_rec,
        "val/f1": val_f1,
        "learning_rate": lr_scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

# 最终测试集评估
print("\n===== 测试集评估 =====")
test_acc, test_prec, test_rec, test_f1 = evaluate(test_loader, model, device)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test F1: {test_f1:.4f}")

# 保存模型
model.save_pretrained("bert_toutiao_model")
tokenizer.save_pretrained("bert_toutiao_model")
print("模型已保存到 bert_toutiao_model 目录")

# 关闭 SwanLab
swanlab.finish()
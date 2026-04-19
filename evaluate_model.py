import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# ========== 1. 数据加载（与之前相同） ==========
def load_toutiao_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                data.append({'text': parts[3], 'label': parts[2]})
    return pd.DataFrame(data)

test_df = load_toutiao_data('test_1k.txt')

unique_labels = sorted(set(test_df['label'].unique()))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(unique_labels)

test_df['label_id'] = test_df['label'].map(label2id)

# ========== 2. 加载模型和分词器 ==========
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 加载你训练好的最佳模型权重
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

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
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}

test_dataset = ToutiaoDataset(test_df['text'].values, test_df['label_id'].values, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========== 4. 推理并收集结果 ==========
all_preds = []
all_labels = []
all_probs = []  # 如果需要概率

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ========== 5. 打印分类报告 ==========
print("\n分类报告（各类别详细指标）:")
report = classification_report(all_labels, all_preds, target_names=unique_labels, zero_division=0)
print(report)

# ========== 6. 绘制并保存混淆矩阵 ==========
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("混淆矩阵图片已保存: confusion_matrix.png")

# ========== 7. 绘制各类别 F1 分数柱状图 ==========
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
class_names = unique_labels

# 按 F1 分数排序以便更好展示
sorted_indices = np.argsort(per_class_f1)
sorted_classes = [class_names[i] for i in sorted_indices]
sorted_f1 = per_class_f1[sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(sorted_classes, sorted_f1, color='skyblue')
plt.xlabel('F1 Score')
plt.title('Per-class F1 Scores on Test Set')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('per_class_f1.png', dpi=300)
plt.close()
print("各类别 F1 柱状图已保存: per_class_f1.png")

# ========== 8. 输出总体指标 ==========
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
print(f"\n总体测试指标:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
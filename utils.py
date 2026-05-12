import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def load_toutiao_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                data.append({'text': parts[3], 'label': parts[2]})
    return pd.DataFrame(data)

def collate_fn(batch, tokenizer, max_len=128):
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    encoding = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'labels': labels}

def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)   # 直接获得 logits 张量
        preds = torch.argmax(logits, dim=1)                            # 去掉 .logits
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds
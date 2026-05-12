import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    """自定义BERT分类模型，符合 nn.Module 规范"""
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 标记的 pooled 输出
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
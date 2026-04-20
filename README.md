# BERT中文文本分类（今日头条数据集）

基于 `bert-base-chinese` 实现的中文新闻标题分类，集成 Focal Loss 和 FGM 对抗训练。

## 实验结果

- 测试集准确率：83.36%（超过参考指标 83%）
- 测试集 F1 分数：0.8341
- 少数类 `stock` F1 从基线 0 提升至 0.73（Focal Loss 有效缓解类别不平衡）

先后进行了四次超参数对比实验：
1. lr=2e-5, bs=32：测试准确率 82.05%
2. lr=1e-5, bs=32：测试准确率 79.32%（学习率过小导致欠拟合）
3. lr=3e-5, bs=32：测试准确率 83.36%（最佳性能之一）
4. lr=3e-5, bs=16：测试准确率 83.36%，验证准确率达 84.40%（验证集表现更优）

结论：学习率 `3e-5` 优于 `2e-5` 和 `1e-5`；批量大小 `16` 与 `32` 测试性能持平，但前者验证集更高。

## 快速开始

```bash
# 环境配置
conda create -n demo1 python=3.9
conda activate demo1
pip install torch transformers swanlab scikit-learn pandas tqdm matplotlib seaborn

# 训练
python train_toutiao.py

# 评估（生成混淆矩阵和 F1 柱状图）
python evaluate_model.py

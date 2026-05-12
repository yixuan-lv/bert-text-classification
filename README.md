# 基于BERT的中文新闻标题分类（标准交叉熵版本）

本项目基于 `bert-base-chinese` 预训练模型，使用自定义 `BertClassifier`（`nn.Module`）对今日头条新闻标题进行15分类。实现了数据预处理、模型训练、超参数对比与可视化，代码结构清晰。

## 数据集

- **来源**: 今日头条客户端（2018年5月采集）
- **下载地址**: [GitHub (aceimnorstuvwxz)](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)
- **类别数量**: 15个
- **数据格式**: `id!_!code!_!category!_!title!_!keywords`
- **数据划分**: train_3k.txt (3000条)、dev_1k.txt (1000条)、test_1k.txt (1064条)

## 项目结构
.
├── config.json # 超参数配置文件
├── train.py # 主训练脚本
├── dataset.py # 数据集类
├── model.py # 自定义BertClassifier模型
├── utils.py # 数据加载、collate_fn、评估函数
├── data/ # 数据集（已忽略上传）
├── requirements.txt # 依赖包列表
├── README.md
└── .gitignore

## 环境配置

```bash
conda create -n bert-text-cls python=3.9
conda activate bert-text-cls
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


requirements.txt 内容：
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.1.0
tqdm>=4.62.0
超参数对比实验
为分析学习率和批次大小对模型性能的影响，设计4组实验（训练3个epoch，使用标准交叉熵损失）：

实验编号	learning_rate	batch_size	测试准确率
1	2e-5	32	81.95%
2	1e-5	32	80.17%
3（最优）	3e-5	32	83.46%
4	3e-5	16	83.46%
结论：

学习率 3e-5 优于 2e-5 和 1e-5，稍大的学习率在此任务上收敛更好。

批次大小 16 与 32 达到相同准确率（83.46%），考虑到训练效率，选择 batch_size=32 作为最终配置。

因此，最优模型为 学习率 3e-5，批次大小 32。

最优模型实验结果（实验三）
整体性能
指标	数值
测试准确率	83.46%
测试F1分数	0.8346
最佳验证准确率	83.20%
各类别详细指标
类别	precision	recall	f1-score	support
news_agriculture	0.84	0.79	0.82	53
news_car	0.91	0.94	0.93	99
news_culture	0.94	0.83	0.88	77
news_edu	0.86	0.92	0.89	74
news_entertainment	0.85	0.88	0.86	108
news_finance	0.72	0.77	0.75	74
news_game	0.76	0.85	0.80	80
news_house	0.85	0.80	0.82	49
news_military	0.83	0.72	0.78	69
news_sports	0.97	0.84	0.90	103
news_story	0.82	0.82	0.82	17
news_tech	0.75	0.82	0.78	114
news_travel	0.85	0.86	0.86	59
news_world	0.75	0.82	0.79	74
stock	0.86	0.43	0.57	14
加权平均	0.84	0.83	0.83	1064
注：stock 类样本极少（14个），F1为0.57，属于数据不平衡导致的正常现象。

可视化结果
混淆矩阵
https://confusion_matrix.png

各类别 F1 分数柱状图
https://per_class_f1.png

运行说明
bash
# 训练、测试并生成图表（默认使用 config.json 中的超参数）
python train.py
运行后将在当前目录生成：

best_model.pth：最佳模型权重

confusion_matrix.png：混淆矩阵热力图

per_class_f1.png：各类别 F1 分数柱状图

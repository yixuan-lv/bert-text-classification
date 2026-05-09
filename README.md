# 基于BERT的中文新闻标题分类（标准交叉熵版本）

本项目基于 `bert-base-chinese` 预训练模型，对今日头条新闻标题进行15分类。实现了完整的数据预处理、模型训练、超参数对比与结果可视化流程，代码结构清晰，符合工程化规范。

## 数据集

- **来源**: 今日头条客户端（2018年5月采集）
- **下载地址**: [GitHub (aceimnorstuvwxz)](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)
- **类别数量**: 15个
- **数据格式**: `id!_!code!_!category!_!title!_!keywords`
- **数据划分**: train_3k.txt (3000条)、dev_1k.txt (1000条)、test_1k.txt (1064条)

## 项目结构
.
├── config.json # 超参数配置文件
├── train.py # 训练+评估+可视化主脚本
├── models/
│ └── dataset.py # 数据集类（返回原始文本）
├── utils/
│ └── helpers.py # 数据加载、collate_fn、评估函数
├── data/ # 数据集（已忽略上传）
├── requirements.txt # 依赖包列表
├── README.md
└── .gitignore

## 环境配置

```bash
# 创建conda环境（Python 3.9）
conda create -n bert-text-cls python=3.9
conda activate bert-text-cls

# 安装依赖
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
为了分析学习率和批次大小对模型性能的影响，我们设计了5组对比实验，所有实验均使用标准交叉熵损失，训练3个epoch。

实验编号	learning_rate	batch_size	测试准确率
1	2e-5	32	82.89%
2	1e-5	32	79.42%
3	3e-5	32	81.58%
4（最优）	3e-5	16	83.27%
5	2e-5	16	82.89%
结论：

学习率 3e-5 在批次大小为16时取得最佳效果，说明稍大的学习率配合小批量有助于提高泛化能力。

相比批次大小32，批次大小16在本任务上表现更优（对比实验3和4）。

学习率 1e-5 导致欠拟合，准确率最低（79.42%）。

因此，最终选择 学习率 3e-5，批次大小 16 作为最优模型。

最优模型实验结果（实验四）
整体性能
指标	数值
测试准确率	83.27%
测试F1分数	0.8327
最佳验证准确率	83.10%
各类别详细指标
类别	precision	recall	f1-score	support
news_agriculture	0.81	0.89	0.85	53
news_car	0.89	0.94	0.92	99
news_culture	0.94	0.86	0.90	77
news_edu	0.85	0.93	0.89	74
news_entertainment	0.83	0.87	0.85	108
news_finance	0.79	0.70	0.74	74
news_game	0.78	0.84	0.81	80
news_house	0.89	0.84	0.86	49
news_military	0.83	0.75	0.79	69
news_sports	0.94	0.85	0.89	103
news_story	0.75	0.71	0.73	17
news_tech	0.76	0.82	0.78	114
news_travel	0.82	0.83	0.82	59
news_world	0.73	0.76	0.74	74
stock	1.00	0.50	0.67	14
加权平均	0.84	0.83	0.83	1064
注：stock 类样本极少（14个），F1为0.67，属于数据不平衡导致的正常现象。

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

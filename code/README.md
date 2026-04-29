# ChemBERTa Binding Energy Prediction - Training Code

## 文件结构

```
code/
├── config.py           # 配置参数
├── dataset.py          # 数据加载和预处理
├── model.py            # 模型定义
├── evaluate.py         # 评估函数
├── train.py            # 训练主脚本
├── smiles_augmentation.py # SMILES随机等价表示增强
├── analyze_results.py  # 消融/增强实验汇总与报告
├── test_local.py       # 本地测试（不需要 GPU）
└── README.md           # 本文件
```

## 快速开始

### 1. 本地测试（不需要 GPU）

在本地验证代码逻辑：

```bash
cd code
python test_local.py
```

这会：
- 创建 dummy 数据
- 测试 dataset 加载
- 测试模型创建
- 测试 forward pass
- 测试训练步骤
- 测试评估函数

如果所有测试通过，说明代码逻辑正确。

### 2. 准备真实数据

确保你有 scaffold split 数据文件：

```
../data/split_seed42.csv
../data/split_seed123.csv
../data/split_seed456.csv
```

每个文件应该包含列：
- `SMILES`: 分子 SMILES 字符串
- `Energy_min`: Binding energy 标签
- `split`: 'train', 'val', 或 'test'

### 3. 在 GPU 环境训练

```bash
cd /hdd2/zesen/daily/5240/5240-project

# 主方法：两阶段监督训练
python code/train.py --experiment main --data-dir data --results-dir results

# 消融：只训练head，不做Stage 2
python code/train.py --experiment stage1_only --data-dir data --results-dir results

# 消融：跳过head warm-up，直接微调encoder/head
python code/train.py --experiment stage2_only --data-dir data --results-dir results

# 对照：标准full fine-tuning
python code/train.py --experiment full_finetune --data-dir data --results-dir results

# 对照：随机初始化encoder
python code/train.py --experiment from_scratch --data-dir data --results-dir results

# 增强实验：训练集on-the-fly随机SMILES
python code/train.py --experiment main --augment --aug-prob 0.5 --data-dir data --results-dir results

# 汇总分析与可视化
python code/analyze_results.py
```

## 配置说明

在 `config.py` 中可以修改：

### 模型配置
- `MODEL_NAME`: HuggingFace 模型名称
- `FREEZE_ENCODER`: True = 主方法（linear probing），False = Ablation 1（full fine-tune）
- `HEAD_HIDDEN_DIM`: Regression head 隐藏层维度
- `HEAD_DROPOUT`: Dropout 比例

### 训练配置
- `BATCH_SIZE`: 批大小（GPU 内存不够可以调小）
- `MAX_EPOCHS`: 最大训练轮数
- `LEARNING_RATE`: 学习率
- `EARLY_STOPPING_PATIENCE`: Early stopping 耐心值

### 数据配置
- `LABEL_COL`: 标签列名（默认 'Energy_min'）
- `SMILES_COL`: SMILES 列名
- `MAX_LENGTH`: SMILES 最大 token 长度
- `AUGMENT_SMILES`: 只在训练集启用SMILES随机化
- `AUGMENTATION_PROB`: 每次读取训练样本时替换为随机等价SMILES的概率

## 方法设计

主方法是两阶段 supervised regression：

1. Stage 1 冻结ChemBERTa encoder，只训练regression head。这个阶段把预训练分子表示映射到当前任务的binding energy尺度，降低小数据集上直接微调全部参数的不稳定性。
2. Stage 2 解冻encoder，并对head/backbone使用不同学习率。head保持较快适配，backbone用更小学习率做任务相关调整，避免破坏预训练化学表示。

SMILES增强只作用于训练集：同一分子的不同合法SMILES共享同一个能量标签。这样不会改变监督信号，而是要求模型对SMILES文本序列的等价重写保持一致，有助于检验模型是否学习分子结构而不是记忆单一字符串表示。

## 输出文件

训练完成后会生成：

```
../results/main_method_seed42/
├── results.json          # 训练历史和指标
├── predictions.csv       # 测试集预测结果
└── best_model.pt         # 最佳模型权重

../results/main_method_summary.csv       # 所有seed或CV fold汇总
../results/main_method_aug0.5_summary.csv # 增强实验汇总，概率写入文件名避免覆盖
../results/experiment_analysis.md        # analyze_results.py生成的方法与结果说明
```

## 修改为 Ablation 实验

### Ablation 1: Full Fine-tune

修改 `config.py`:
```python
FREEZE_ENCODER = False
LEARNING_RATE = 2e-5  # 更小的学习率
```

### Ablation 2: From Scratch

修改 `model.py` 中的 `ChemBERTaRegressor.__init__`:
```python
# 不加载预训练权重
from transformers import RobertaConfig, RobertaModel

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)
self.encoder = RobertaModel(config)  # 随机初始化
```

## 依赖安装

```bash
pip install torch transformers pandas scikit-learn numpy
```

如果需要 GPU：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 常见问题

### Q: CUDA out of memory
A: 减小 `BATCH_SIZE`，比如改成 8 或 4

### Q: 训练太慢
A: 
- 检查是否在用 GPU（`config.DEVICE` 应该是 'cuda'）
- 增加 `num_workers` 在 dataloader 里

### Q: 模型不收敛
A:
- 检查 label 是否正确标准化
- 尝试调整学习率
- 检查数据是否有问题

### Q: 想看训练过程
A: 设置 `config.VERBOSE = True`

## 下一步

训练完成后：
1. 查看 `results.json` 了解训练历史
2. 查看 `predictions.csv` 分析预测结果
3. 用 `main_method_summary.csv` 报告seed或CV平均结果
4. 运行 `python code/analyze_results.py` 生成 `experiment_analysis.md`、`ablation_comparison.png` 和 `augmentation_comparison.png`

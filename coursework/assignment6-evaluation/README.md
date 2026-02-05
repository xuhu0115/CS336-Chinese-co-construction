# CS336 Assignment 6: 大型语言模型评测框架介绍

本文件夹包含CS336课程第六次作业的内容，主要介绍常用的几种大型语言模型评测框架及其使用方法。

## 📁 文件夹结构

```
assignment6_evaluation/
├── demo.ipynb                    # 主要的演示notebook
├── lm_eval_demo.py              # lm-evaluation-harness 极简实现脚本
├── evalscope_demo.py            # evalscope 极简实现脚本
├── data/                        # 数据文件夹
│   └── index_testset.jsonl      # evalscope生成的评测数据集
├── outputs/                     # 评测输出结果
│   ├── 20260119_232050/         
│   └── 20260120_000654/         
├── images/                      # 图片
│   └── evalscope_panel.png      
├── lm-evaluation-harness/       # lm-evaluation-harness 框架源码
└── README.md                    # 本文件
```

## 🎯 作业目标

介绍常用的几种大型语言模型评测框架：
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - 学术界标准评测框架
- [evalscope](https://github.com/modelscope/evalscope) - 支持自定义数据集组合和可视化分析
- [Evalchemy](https://github.com/mlfoundations/evalchemy) - 轻量级评测框架
- [lighteval](https://github.com/huggingface/lighteval) - Hugging Face生态集成

重点演示**lm-evaluation-harness**和**evalscope**的使用方法。

## 📊 评测框架对比

| 框架名称 | 开发机构 | 主要特点 | 适用场景 |
|---------|---------|---------|---------|
| lm-evaluation-harness | EleutherAI | 功能丰富，支持多种模型和任务，学术界标准 | 学术研究、基准测试 |
| evalscope | ModelScope | 支持自定义数据集组合，可视化分析，中文友好 | 产业应用、模型评估 |
| Evalchemy | ML Foundations | 轻量级，注重可复现性和扩展性 | 研究实验、快速原型 |
| lighteval | Hugging Face | 集成Transformers生态，易于使用 | Hugging Face用户 |

## 🔧 主要内容

### lm-evaluation-harness

- **零样本评测**：arc_easy, piqa, lambada, triviaqa
- **少样本评测**：humaneval, mbpp, gsm8k, minerva_math
- **多维度能力评测**：通用语言理解、常识推理、代码、数学推理

### evalscope

- **自定义数据集组合**：通过CollectionSchema定义评测索引
- **加权采样**：根据业务需求调整数据集权重
- **可视化分析**：通过Web界面分析评测结果详情


## 🚀 快速开始

### 环境准备

```bash
# 1. 创建并激活conda环境
conda create -n eval_env python=3.10
conda activate eval_env

# 2. 安装lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install -e .[math]

# 3. 安装evalscope
pip install evalscope
pip install 'evalscope[app]' -U  # 可视化依赖
```

### 开始学习

进入 `demo.ipynb` 跟着学习两种框架的简单使用，更详细内容可参考框架的指导手册。

如果你想直接运行：

#### 1. lm-evaluation-harness演示

```bash
# Python脚本方式
python lm_eval_demo.py
```

#### 2. evalscope演示

```python
# 运行evalscope演示
python evalscope_demo.py
```

## 📝 使用建议

- **学术研究**：推荐使用 `lm-evaluation-harness`
- **产业应用**：推荐使用 `evalscope`
- **快速原型**：推荐使用 `Evalchemy`
- **Hugging Face用户**：推荐使用 `lighteval`

## 📞 更多内容

如有问题，请参考：
- [lm-evaluation-harness文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [evalscope文档](https://github.com/modelscope/evalscope)

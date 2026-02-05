<div align='center'>
    <img src="./images/head.jpg" alt="alt text" width="100%">
    <h1>Diy-LLM</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/xuhu0115/diy-llm?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/xuhu0115/diy-llm?style=flat&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/xuhu0115/diy-llm"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
</div>

<div align="center">
  <p><a href="https://github.com/xuhu0115/diy-llm">📚 在线阅读地址</a></p>
  <h3>📚 带你系统性学习大语言模型</h3>
  <p><em>一座为中文学习者量身打造的"LLM炼丹工坊"</em></p>
</div>

我们希望这门 CS336 中文课程，不只是斯坦福原版的"汉化版"，而是一座为中文学习者量身打造的"LLM炼丹工坊"。在这里，你亲手锻造理解、打磨代码、调控火候，最终炼出属于自己的大模型真丹。

## 📋 前置要求

- **Python 编程**：熟练掌握 Python 和软件工程能力
- **深度学习基础**：熟悉 PyTorch，了解神经网络基本原理
- **数学基础**：线性代数、概率统计、微积分
- **机器学习**：需对机器学习与深度学习的基础知识有扎实掌握
- **GPU 编程（可选）**：了解 CUDA 基础概念会更佳，不懂也没关系，本项目也有入门教程

## 📚 课程愿景

- **硬核理论与动手实战并重**：我们会完整保留原版课程的技术深度，但会用更符合中文学习者思维的方式重构知识体系。对于必要的数学、深度学习前置知识，我们会帮你补齐，确保学习曲线平滑，让每个认真投入的人都能跟上。
- **搭建一套循序渐进的知识体系**：将构建LLM这个庞大工程，拆解成一个个可以上手、可以理解的模块。学完后，你将拥有一个关于LLM的完整知识图谱。
- **代码驱动，知行合一**：课程的核心是"用代码思考"。所有作业，我们不仅会提供实现代码，更会分享写下每一行代码时的思考过程。
- **贴近国内环境的本土化改造**：考虑到国内的网络环境、大家手头的计算资源以及独特的开源生态，我们会提供更接地气的解决方案和案例（比如，多聊聊Qwen、DeepSeek等国产优秀模型）。

## 🎯 项目意义

学完这门课，你能得到什么？

- **扎实的技术地基**：你将能亲手"造"出自己的LLM，对每个核心组件都了然于胸。
- **宝贵的工程经验**：掌握从数据处理、模型训练到部署优化，堪比大厂的全流程实战技能。
- **突出的行业竞争力**：具备大模型研发的核心能力，为你进入心仪的大厂或团队铺平道路。
- **清晰的科研视野**：对LLM领域有体系化的认知，为未来深入研究打下坚实的基础。

## 📖 课程目录

| 章节 | 关键内容 | 配套作业 | 状态 |
|------|----------|----------|------|
| [前言](docs/前言.md) | 项目由来、背景及学习建议 | - | ✅ |
| [第1章 工具使用](docs/chapter1) | W&B 使用与实验追踪 | - | ✅ |
| [第2章 分词器](docs/chapter2/chapter2_分词器.md) | 分词器原理与 BPE 实现 | [作业1](coursework/assignment1-basics/) | ✅ |
| [第3章 PyTorch 与资源核算](docs/chapter3/) | 训练原语、算力/显存估算 | - | ✅ |
| [第4章 语言模型架构与训练细节](docs/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md) | Transformer 架构与训练要点 | [作业1](coursework/assignment1-basics/) | ✅ |
| [第5章 混合专家模型](docs/chapter5/chapter5_混合专家模型.md) | MoE 原理、路由与工程实践 | - | ✅ |
| [第6章 GPU 与相关优化](docs/chapter6/chapter6_第六章GPU和GPU相关的优化.md) | GPU 基础与优化技巧 | [作业2](coursework/assignment2-systems/) | ✅ |
| [第7章 GPU 高性能编程](docs/chapter7/chapter7_第七章GPU高性能编程.md) | CUDA 与高性能编程 | [作业2](coursework/assignment2-systems/) | ✅ |
| [第8章 分布式训练](docs/chapter8/chapter8_第八章分布式训练.md) | 并行范式与跨机训练 | [作业2](coursework/assignment2-systems/) | ✅ |
| [第9章 Scaling Laws](docs/chapter9/chapter9_Scaling_Laws.md) | 扩展定律与实验 | [作业3](coursework/assignment3-scaling/) | ✅ |
| [第10章 推理](docs/chapter10/推理.md) | 推理性能与落地优化 | [作业6](coursework/assignment6-evaluation/) | ✅ |
| [第11章 数据工程](docs/chapter11/chapter11_数据工程.md) | 数据清洗、构建与管理 | [作业4](coursework/assignment4-data/) | ✅ |
| [第12章 评估与基准测试](docs/chapter12/chapter12_评估与基准测试.md) | 指标体系与评测方法 | [作业6](coursework/assignment6-evaluation/) | ✅ |
| [第13章 大模型的基本训练流程](docs/chapter13/chapter13_第十三章大模型的基本训练流程.md) | 预训练、SFT、RL 流程 | [作业5](coursework/assignment5-alignment/) | ✅ |
| [第14章 可验证奖励的强化学习](docs/chapter14/chapter14_可验证奖励的强化学习.md) | RLVR 思想与实践 | [作业5](coursework/assignment5-alignment/) | ✅ |
| [第15章 扩展内容](docs/chapter15/) | - | - | 🚧 |

> 状态图例说明：✅ 已完成  🔄 更新中	📝 待完善	🚧 筹备中	 ⏸️ 暂缓	


## 📝 作业概览

| 作业 | 核心任务 |状态 |
|------|----------|------|
| [作业1：手搓大模型](coursework/assignment1-basics/) | 实现 tokenizer、model architecture、optimizer，训练一个极简语言模型 | ✅ |
| [作业2：系统优化](coursework/assignment2-systems/) | 性能分析与基准测试；用 Triton 实现 FlashAttention-2；构建分布式训练代码 | ✅ |
| [作业3：扩展定律](coursework/assignment3-scaling/) | 理解 Transformer 各组件功能；拟合 scaling law 预测模型扩展效果 | ✅ |
| [作业4：数据处理](coursework/assignment4-data/) | 将 Common Crawl 原始数据转换为预训练数据集，执行过滤与去重 | ✅ |
| [作业5：模型对齐](coursework/assignment5-alignment/) | 应用 SFT 与强化学习（如 GRPO）训练模型解决数学问题 | ✅ |
| [作业6：模型评估](coursework/assignment6-evaluation/) | 使用 lm-evaluation-harness 和 evalscope 进行多维度评测（语言理解、常识推理、代码、数学推理） | ✅ |


## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/xuhu0115/diy-llm.git
cd CS336-Chinese-co-construction
# 安装基础依赖（根据具体作业需求安装）
```

### 学习路径

1️⃣ 理论学习 → 按章节顺序阅读 `docs/` 目录下的文档
2️⃣ 实践练习 → 完成 `coursework/` 目录下的 6 个作业
3️⃣ 深入理解 → 阅读代码实现，理解每个组件的设计

### 项目结构

```
CS336-Chinese-co-construction/
├── docs/                    # 理论章节文档
│   ├── 前言/           
│   ├── chapter1/           # 工具使用
│   ├── chapter2/           # 分词器
│   ├── chapter3/           # PyTorch 与资源核算
│   ├── chapter4/           # 语言模型架构与训练细节
│   ├── chapter5/           # 混合专家模型
│   ├── chapter6/           # GPU 与相关优化
│   ├── chapter7/           # GPU 高性能编程
│   ├── chapter8/           # 分布式训练
│   ├── chapter9/           # Scaling Laws
│   ├── chapter10/          # 推理
│   ├── chapter11/          # 数据工程
│   ├── chapter12/          # 评估与基准测试
│   ├── chapter13/          # 大模型的基本训练流程
│   └── chapter14/          # 可验证奖励的强化学习
├── coursework/              # 实践作业
│   ├── assignment1-basics/        # 作业1：手搓大模型
│   ├── assignment2-systems/       # 作业2：系统优化
│   ├── assignment3-scaling/       # 作业3：扩展定律
│   ├── assignment4-data/          # 作业4：预训练数据处理
│   ├── assignment5-alignment/     # 作业5：对齐
│   └── assignment6-evaluation/    # 作业6：评估
├── README.md               # 项目说明
└── .gitignore              # Git忽略配置
```

## 🔗 相关链接

- **仓库地址**：https://github.com/xuhu0115/diy-llm
- **原版课程主页**：[Stanford CS336 (Spring 2025)](https://stanford-cs336.github.io/spring2025/)
- **原版课程项目**：https://github.com/stanford-cs336/spring2025-lectures/tree/main

## ❓ 常见问题

<details>
<summary><b>Q: 没有 GPU 可以学习吗？</b></summary>

理论部分可以正常学习，作业中的部分内容可以在 CPU 上调试，但完整训练需要 GPU。建议使用云服务平台。
</details>

<details>
<summary><b>Q: 与原版 CS336 有什么区别？</b></summary>

我们在保留原版技术深度的基础上，针对中文学习者进行了本土化改造，包括中文讲解、作业实现、更详细的参考内容来源、国产模型案例等。
</details>

## 👥 贡献者

### 贡献者名单

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/xuhu0115"><img width="70" height="70" src="https://github.com/xuhu0115.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/qiwang067">徐虎</a> 
        <p>项目负责人<br> Datawhale 成员<br> 上海交通大学 <br> 负责内容：第1、3、9、12、14章；作业5、6；全文内容审核</p>
      </td>
      <td>
         <a href="https://github.com/kangkang-Adam"><img width="70" height="70" src="https://github.com/kangkang-Adam.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/kangkang-Adam">x</a> 
        <p>项目负责人<br> x<br> x <br> 负责内容：第4、6、7、8、13章；作业2、4</p>
      </td>
      <td>
         <a href="https://github.com/1iyouzhen"><img width="70" height="70" src="https://github.com/1iyouzhen.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/1iyouzhen">x</a>
         <p>项目负责人<br> Datawhale-鲸英助教 <br>负责内容：第2、5、10、11、13章；作业1、3； </p>
      </td>
    </tr>
  </tbody>
</table>

*注：我们感谢每一位为项目做出贡献的开发者！*

我们欢迎所有形式的贡献！无论是文档改进、代码优化、bug修复还是新内容添加，都是对项目的宝贵支持。

### 如何贡献

1. **报告问题**：如果发现文档错误、代码bug或改进建议，欢迎提交 [Issue](https://github.com/xuhu0115/diy-llm/issues)
2. **提交代码**：Fork 本仓库，创建你的特性分支，提交更改后发起 Pull Request
3. **完善文档**：帮助改进文档、翻译内容或添加示例
4. **分享经验**：在讨论区分享学习心得和实践经验

### 贡献指南

- 提交代码前请确保代码风格一致
- 添加新内容时请遵循现有的文档格式
- 提交PR时请提供清晰的描述和变更说明
- 欢迎在Issue中讨论大的改动方案



## 📝 更新日志

项目持续更新中，最新进展请查看 [GitHub Releases](https://github.com/xuhu0115/diy-llm/releases) 或提交记录。

## 📄 许可证

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。

## 🙏 致谢

- 感谢 Stanford CS336 课程团队提供优秀的原版课程
- 特别感谢[@Sm1les](https://github.com/Sm1les)对本项目的帮助与支持
- 感谢所有为项目做出贡献的开发者
- 感谢开源社区的支持与反馈

## ⭐ Star History

如果这个项目对你有帮助，欢迎给个 Star ⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=xuhu0115/diy-llm&type=Date)](https://star-history.com/#xuhu0115/diy-llm&Date)

---

<div align="center">
  <p>让更多人能够系统性地学习大语言模型构建技术</p>
  <p>Made with ❤️ by the Datawhale</p>
</div>

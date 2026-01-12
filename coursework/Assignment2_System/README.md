# CS336 2025年春季学期作业2：系统

有关作业的完整描述，请参见作业文档
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

如果您发现作业文档或代码中存在任何问题，请随时提出GitHub issue或提交修复的pull request。

## 环境设置

本目录组织结构如下：

- [`./cs336-basics`](./cs336-basics)：包含模块`cs336_basics`及其相关`pyproject.toml`的目录。该模块包含了作业1中语言模型的官方实现。如果您想使用自己的实现，可以将此目录替换为您自己的实现。
- [`./cs336_systems`](./cs336_systems)：此文件夹基本上是空的！这是您将实现优化后的Transformer语言模型的模块。您可以随意从作业1（位于`cs336-basics`）中获取所需的代码并复制过来作为起点。此外，您还将在本模块中实现分布式训练和优化。

直观来看，目录结构应类似于：

``` sh
.
├── cs336_basics  # 名为cs336_basics的Python模块
│   ├── __init__.py
│   └── ... cs336_basics模块中的其他文件，来自作业1 ...
├── cs336_systems  # TODO(你)：你为作业2编写的代码 
│   ├── __init__.py
│   └── ... TODO(你)：作业2所需的任何其他文件或文件夹 ...
├── README.md
├── pyproject.toml
└── ... TODO(你)：作业2所需的其他文件或文件夹 ...
```

如果您想使用自己作业1的实现，请将`cs336-basics`目录替换为您自己的实现，或编辑外层`pyproject.toml`文件以指向您自己的实现。

0. 我们使用`uv`管理依赖项。您可以通过运行以下命令来验证`cs336-basics`包中的代码是否可访问：

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run`会根据`pyproject.toml`文件自动安装依赖项。

## 提交作业

要提交作业，请运行`./test_and_make_submission.sh`。该脚本将安装代码的依赖项、运行测试并创建一个gzip压缩的tar包。我们应该能够解压您提交的tar包并运行`./test_and_make_submission.sh`来验证您的测试结果。

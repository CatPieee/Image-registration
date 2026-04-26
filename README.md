# Image Registration (图像配准项目)

这是一个关于图像配准的学习与研究项目，主要使用 Kornia 和 PyTorch 实现基于 LoFTR (Local Feature Transformer) 的特征匹配。

## 项目结构

- `main.py`: 核心运行脚本，支持批量处理多个图像对。它会自动应用随机变换，并对比 **MatchAnything (SOTA)**, **LoFTR** 和 **SIFT** 的配准效果。
- `chart_generator.py`: 自动读取实验数据并生成专业对比图表。
- `data/`: 存放数据集的目录，包含 TNO 图像融合数据集等。
- `output/`: 自动生成的目录，存放配准结果图、匹配可视化图及实验统计图表。
- `requirements.txt`: 项目所需的 Python 依赖包。

## 数据集说明

本项目使用了 **TNO Image Fusion Dataset**。

- **数据集来源**: [yanyanchun/TNO_Image_Fusion_Dataset](https://github.com/yanyanchun/TNO_Image_Fusion_Dataset)
- **数据准备**: 
    1. 访问上述 GitHub 链接并下载数据集。
    2. 在项目根目录下创建 `data/` 文件夹。
    3. 将下载的数据集解压并放入 `data/` 目录下，确保路径结构如：`data/TNO_Image_Fusion_Dataset-master/...`。

> **注意**: `data/` 文件夹已被添加到 `.gitignore` 中，不会被 Git 追踪。

## 环境配置与安装

请按照以下步骤配置开发环境：

### 1. 创建虚拟环境

在项目根目录下运行以下命令创建 Python 虚拟环境：

```bash
python -m venv .venv
```

### 2. 激活虚拟环境

- **macOS / Linux:**

```bash
source .venv/bin/activate
```

- **Windows:**

```bash
.venv\Scripts\activate
```

### 3. 安装依赖包

激活虚拟环境后，使用 pip 安装所需的库：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 运行配准实验
目前项目的主要逻辑在 `main.py` 中。你可以通过以下命令运行批量配准实验：

```bash
python main.py
```

该脚本会自动遍历预设的图像对，应用随机变换，并对比 LoFTR 与 SIFT 的配准表现。

### 2. 生成分析图表
运行完 `main.py` 后，执行以下命令生成定量分析图表：

```bash
python chart_generator.py
```

> **注意**: 
> 1. 运行前请确保 `data/` 目录下已准备好 TNO 数据集。
> 2. 配准结果、可视化连线图以及统计图表（如 `performance_chart.png`）都会自动保存到 `output/` 文件夹中。

## 依赖库说明

- **PyTorch**: 深度学习框架。
- **Kornia**: 基于 PyTorch 的计算机视觉库，用于图像特征提取与匹配。
- **OpenCV**: 用于图像处理与可视化。

# Image Registration (图像配准项目)

这是一个关于图像配准的学习与研究项目，主要使用 Kornia 和 PyTorch 实现基于 LoFTR (Local Feature Transformer) 的特征匹配。

## 项目结构

- `main.py`: 项目核心入口，演示了如何使用 LoFTR 进行图像配准及交互式预览。
- `data/`: 存放数据集的目录，包含 TNO 图像融合数据集等。
- `output/`: 自动生成的目录，存放配准后的结果图和匹配可视化图。
- `requirements.txt`: 项目所需的 Python 依赖包。

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

目前项目的主要逻辑在 `main.py` 中。你可以通过以下命令运行示例：

```bash
python main.py
```

> **注意**: 
> 1. 运行前请确保 `main.py` 中指定的图像路径存在。
> 2. 运行后会弹出一个交互式窗口，你可以拖动上方的 **Alpha** 滑动条来实时调节两张图像的叠加透明度，从而直观验证配准效果。
> 3. 配准结果会自动保存到 `output/` 文件夹中。

## 依赖库说明

- **PyTorch**: 深度学习框架。
- **Kornia**: 基于 PyTorch 的计算机视觉库，用于图像特征提取与匹配。
- **OpenCV**: 用于图像处理与可视化。

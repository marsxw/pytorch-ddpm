# PyTorch 实现 DDPM (基于 MNIST 数据集)

这是一个基于 PyTorch 实现的 **去噪扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPM)**。本项目通过解析原始 MNIST 二进制数据集进行训练，能够从纯高斯噪声中生成清晰的手写数字图像。

---

## 1. 核心原理

### 前向扩散过程 (Forward Process)
在前向过程中，我们按照预定义的线性 Beta 调度 ($T=1000$) 逐步向原始图像添加高斯噪声。随着步数 $t$ 的增加，图像特征逐渐消失，最终变成纯噪声。

**数学公式:**

$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I}) $$

### 反向去噪过程 (Backward Process)
神经网络（U-Net）学习预测在特定时间步 $t$ 添加到图像中的噪声。采样时，模型从纯噪声开始，通过 1000 次迭代逐步剔除预测噪声，从而还原出真实的数字图像。

**采样公式:**

$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z $$

---

## 2. 模型架构

本项目采用了增强版的 **U-Net** 结构，包含以下核心技术：

*   **正弦位置编码 (Sinusoidal Position Embeddings)**: 将标量时间步 $t$ 映射为 256 维向量，让模型理解当前的去噪进度。
*   **残差块 (ResNet Blocks)**: 结合了 **GroupNorm** 归一化和 **SiLU** 激活函数，利用残差连接确保深层网络梯度稳定。
*   **自注意力机制 (Self-Attention)**: 在 U-Net 最底层的特征图 (7x7) 中引入注意力机制，帮助模型捕捉数字的整体拓扑结构。
*   **跳跃连接 (Skip Connections)**: 将编码层特征与解码层拼接，保留图像的局部细节。

---

## 3. 维度流转说明

以 **Batch Size = 128** 为例：

1.  **输入图像**: `[128, 1, 28, 28]`
2.  **时间嵌入**: `[128, 256]`
3.  **U-Net 瓶颈层**: `[128, 256, 7, 7]`
4.  **预测噪声输出**: `[128, 1, 28, 28]` (维度与输入一致)

---

## 4. 训练结果展示

经过 50 个 Epoch 的训练，模型生成的数字清晰且具有多样性：

![生成的样本](sample_epoch_49.png)

---

## 5. 快速上手

### 数据集准备
请确保 MNIST 数据集存放在项目根目录下的 `archive/` 文件夹中：
- `archive/train-images.idx3-ubyte`
- `archive/train-labels.idx1-ubyte`

### 运行程序
```bash
python ddpm_mnist.py
```

### 依赖库
- torch (建议支持 CUDA)
- numpy
- tqdm (训练进度条)
- matplotlib (采样图像保存)

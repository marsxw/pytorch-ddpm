import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import struct
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据加载 (MNIST idx 格式解析)
# ==========================================


class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.transform = transform

    def load_images(self, path):
        # 解析 MNIST 原始二进制文件
        with open(path, 'rb') as f:
            # 前16字节是魔数、图像数量、行数、列数
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_labels(self, path):
        with open(path, 'rb') as f:
            # 前8字节是魔数和标签数量
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # 归一化到 [-1, 1]，这是扩散模型训练的标准范围
        # (0, 255) -> (-1, 1)
        image = (image.astype(np.float32) / 127.5) - 1.0
        # 增加通道维度: [28, 28] -> [1, 28, 28]
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        return image

# ==========================================
# 2. 神经网络组件 (U-Net)
# ==========================================


class SinusoidalPositionEmbeddings(nn.Module):
    """
    DDPM 的 U-Net 在每一个时间步 t 都要共享参数。
    为了让模型知道现在是在处理“全是噪声的第 999 步”还是“快要清晰的第 1 步”，
    我们需要把标量 t 映射成一个具有几何结构的特征向量。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 最终生成的向量维度，例如 256

    def forward(self, time):
        # time 维度: [Batch_Size] (比如 128 个图像，每个图像对应一个不同的 t)
        device = time.device
        half_dim = self.dim // 2

        # 1. 计算频率缩放因子
        # 这个公式来源于 Transformer 论文：10000^(2i/dim)
        # 这里使用 log 空间计算是为了数值稳定性
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # 2. 将时间步 t 与频率因子相乘
        # time[:, None] 维度: [Batch_Size, 1]
        # embeddings[None, :] 维度: [1, half_dim]
        # 相乘后维度: [Batch_Size, half_dim]
        embeddings = time[:, None] * embeddings[None, :]

        # 3. 分别计算正弦和余弦并拼接
        # 这样可以保证生成的向量在空间中具有平滑的周期性，模型更容易学习
        # 最终维度: [Batch_Size, dim] (即 half_dim * 2)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """
    残差块：融合图像特征和时间嵌入。
    输入 x: [B, in_ch, H, W]
    输入 t: [B, time_emb_dim]
    输出: [B, out_ch, H, W]
    """

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        # 第一层卷积
        h = self.gnorm1(self.relu(self.conv1(x)))
        # 时间嵌入：将 [B, C] 变换为 [B, C, 1, 1] 以便与特征图相加
        time_emb = self.relu(self.time_mlp(t))[(..., ) + (None, ) * 2]
        h = h + time_emb
        # 第二层卷积
        h = self.gnorm2(self.relu(self.conv2(h)))
        # 残差连接
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """
    自注意力块：捕捉图像的全局结构。
    输入/输出维度: [B, C, H, W] (保持不变)
    """

    def __init__(self, channels):
        super().__init__()
        self.gnorm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.gnorm(x)
        # 展平 H, W: [B, C, H*W]
        q = self.q(h).view(B, C, -1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        # 计算注意力权重: [B, H*W, H*W]
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # 加权求和并还原维度
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.proj(out)


class ImprovedUNet(nn.Module):
    """
    改进版 U-Net：包含下采样、中间层（带注意力）和上采样。
    """

    def __init__(self):
        super().__init__()
        self.time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU()
        )

        self.init_conv = nn.Conv2d(1, 64, 3, padding=1)

        # 下采样层 (Downsampling)
        self.down1 = ResnetBlock(64, 64, self.time_dim)
        self.down2 = ResnetBlock(64, 128, self.time_dim)
        self.down3 = ResnetBlock(128, 256, self.time_dim)
        self.pool = nn.MaxPool2d(2)

        # 中间层 (Middle)
        self.mid1 = ResnetBlock(256, 256, self.time_dim)
        self.mid_attn = AttentionBlock(256)
        self.mid2 = ResnetBlock(256, 256, self.time_dim)

        # 上采样层 (Upsampling)
        # 注意：输入通道 = 上一层输出 + 跳跃连接(Skip Connection)的通道
        self.up1 = ResnetBlock(384, 128, self.time_dim)  # 256 + 128 = 384
        self.up2 = ResnetBlock(192, 64, self.time_dim)   # 128 + 64 = 192
        self.up3 = ResnetBlock(128, 64, self.time_dim)   # 64 + 64 = 128

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x, timestep):
        # 1. 时间嵌入: [B] -> [B, 256]
        t = self.time_mlp(timestep)

        # 2. 下采样阶段
        x1 = self.init_conv(x)         # [B, 64, 28, 28]
        x2 = self.down1(x1, t)         # [B, 64, 28, 28] (保存用于跳跃连接)

        x3 = self.pool(x2)             # [B, 64, 14, 14]
        x3 = self.down2(x3, t)         # [B, 128, 14, 14] (保存用于跳跃连接)

        x4 = self.pool(x3)             # [B, 128, 7, 7]
        x4 = self.down3(x4, t)         # [B, 256, 7, 7]

        # 3. 中间阶段
        x = self.mid1(x4, t)           # [B, 256, 7, 7]
        x = self.mid_attn(x)           # [B, 256, 7, 7]
        x = self.mid2(x, t)            # [B, 256, 7, 7]

        # 4. 上采样阶段 (使用插值 + 跳跃连接)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 256, 14, 14]
        x = torch.cat([x, x3], dim=1)                        # [B, 384, 14, 14] (拼接)
        x = self.up1(x, t)                                   # [B, 128, 14, 14]

        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 128, 28, 28]
        x = torch.cat([x, x2], dim=1)                        # [B, 192, 28, 28] (拼接)
        x = self.up2(x, t)                                   # [B, 64, 28, 28]

        x = torch.cat([x, x1], dim=1)                        # [B, 128, 28, 28] (拼接)
        x = self.up3(x, t)                                   # [B, 64, 28, 28]

        # 5. 输出层: [B, 1, 28, 28] (预测的噪声)
        return self.out_conv(x)

# ==========================================
# 3. DDPM 调度器 (核心数学逻辑)
# ==========================================


class DDPM:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.T = T
        self.device = device
        # Beta 调度：控制每一步添加多少噪声
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1. - self.betas
        # Alpha 累乘：用于直接从 x_0 计算 x_t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_diffusion(self, x_0, t):
        """
        前向过程 (加噪): q(x_t | x_0)
        公式: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # 返回加噪后的图像 x_t 和 所添加的真实噪声 noise
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def sample(self, model, n_samples=16):
        """
        反向过程 (采样/去噪): p(x_{t-1} | x_t)
        从纯噪声开始，逐步剔除模型预测的噪声。
        """
        model.eval()
        # 从纯高斯噪声开始: [n_samples, 1, 28, 28]
        x = torch.randn((n_samples, 1, 28, 28)).to(self.device)

        for i in reversed(range(self.T)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            # 模型预测噪声
            predicted_noise = model(x, t)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0  # 最后一步不加随机噪声

            # DDPM 采样公式 (算法 2 第 4 步):
            # 数学公式: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * epsilon_theta) + sigma_t * z
            # 维度流转: [16, 1, 28, 28] = 系数 * ([16, 1, 28, 28] - 系数 * [16, 1, 28, 28]) + [16, 1, 28, 28]
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise

        # 还原到 [0, 1] 范围用于显示
        x = (x.clamp(-1, 1) + 1) / 2
        return x

# ==========================================
# 4. 训练流程
# ==========================================


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据准备
    images_path = "archive/train-images.idx3-ubyte"
    labels_path = "archive/train-labels.idx1-ubyte"
    dataset = MNISTDataset(images_path, labels_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 模型初始化
    model = ImprovedUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ddpm = DDPM(T=1000, device=device)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader)
        loss_list = []
        for batch in pbar:
            optimizer.zero_grad()

            # x_0: [B, 1, 28, 28] (原始干净图像)
            x_0 = batch.to(device)
            # 随机采样时间步 t: [B]
            t = torch.randint(0, ddpm.T, (x_0.shape[0],), device=device).long()

            # 生成加噪图像 x_t 和 真实噪声 noise
            x_t, noise = ddpm.forward_diffusion(x_0, t)

            # 模型尝试预测噪声
            predicted_noise = model(x_t, t)

            # 损失函数：预测噪声与真实噪声的 MSE
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            pbar.set_description(f"Epoch {epoch} | Loss: {np.mean(loss_list):.4f}")

        # 定期采样观察效果
        if (epoch + 1) % 5 == 0:
            samples = ddpm.sample(model)
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i].cpu().squeeze(), cmap="gray")
                plt.axis("off")
            plt.savefig(f"sample_epoch_{epoch}.png")
            plt.close()

    # 保存训练好的模型
    torch.save(model.state_dict(), "ddpm_mnist_improved.pth")


if __name__ == "__main__":
    train()

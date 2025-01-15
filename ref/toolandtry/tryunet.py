import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, n=64):  # `n` 是 Unet 中间层通道数，默认为 64
        super(Decoder, self).__init__()
        
        # 1D卷积降维，从576降到1
        self.conv1d_dim_reduce = nn.Conv1d(in_channels=576, out_channels=1, kernel_size=1)
        
        # 下采样部分
        self.down_conv1 = nn.Conv2d(1, n, kernel_size=4, stride=2, padding=1)  # [150, 150] -> [75, 75]
        self.down_conv2 = nn.Conv2d(n, n * 2, kernel_size=4, stride=2, padding=1)  # [75, 75] -> [37, 37]
        self.down_conv3 = nn.Conv2d(n * 2, n * 4, kernel_size=4, stride=2, padding=1)  # [37, 37] -> [18, 18]
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(n * 4, n * 8, kernel_size=3, padding=1)  # [18, 18] -> [18, 18]
        
        # 全连接层，映射到目标大小
        self.fc_down_to_middle = nn.Linear(18 * 18 * n * 8, 45 * 90 * n)
        
        # 上采样部分
        self.upconv1 = nn.ConvTranspose2d(n, n * 4, kernel_size=4, stride=2, padding=1)  # [45, 90] -> [90, 180]
        self.bn1 = nn.BatchNorm2d(n * 4)
        self.conv1_1 = nn.Conv2d(n * 4, n * 4, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(n * 4)
        self.conv1_2 = nn.Conv2d(n * 4, n * 4, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(n * 4)
        
        self.upconv2 = nn.ConvTranspose2d(n * 4, n * 2, kernel_size=4, stride=2, padding=1)  # [90, 180] -> [180, 360]
        self.bn2 = nn.BatchNorm2d(n * 2)
        self.conv2_1 = nn.Conv2d(n * 2, n * 2, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(n * 2)
        self.conv2_2 = nn.Conv2d(n * 2, n * 2, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(n * 2)
        
        self.upconv3 = nn.ConvTranspose2d(n * 2, 1, kernel_size=4, stride=2, padding=1)  # [180, 360] -> [360, 720]
        self.bn3 = nn.BatchNorm2d(1)
        self.conv3_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(1)
        self.conv3_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(1)
        
        # 最终 1x1 卷积
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

        # Skip connections 的线性映射
        # self.fc_skip1 = nn.Linear(75 * 75 * n, 180 * 360 * n * 2)
        # self.fc_skip2 = nn.Linear(37 * 37 * n * 2, 90 * 180 * n * 4)
        # self.fc_skip3 = nn.Linear(18 * 18 * n * 4, 45 * 90 * n)

    def forward(self, x, n):
        # 1D卷积将维度从576降到1
        x = x.permute(0, 2, 1)  # 重排为 (batch, dim, length)
        x = self.conv1d_dim_reduce(x)  # (batch, 1, 22500)
        x = x.permute(0, 2, 1)  # 重排回 (batch, length, dim)
        x = x.view(x.size(0), 1, 150, 150)  # 重塑为 (batch, 1, 150, 150)
        
        # 下采样部分
        skip1 = self.down_conv1(x)  # (batch, n, 75, 75)
        skip2 = self.down_conv2(skip1)  # (batch, n*2, 37, 37)
        skip3 = self.down_conv3(skip2)  # (batch, n*4, 18, 18)
        
        # Bottleneck
        x = self.bottleneck_conv(skip3)  # (batch, n*8, 18, 18)
        
        # 展平并通过全连接层，映射到中间变量大小
        x_flat = x.view(x.size(0), -1)  # 展平为 (batch, 18*18*n*8)
        x_middle = self.fc_down_to_middle(x_flat)  # 映射到 (batch, 45*90*n)
        x_middle = x_middle.view(x.size(0), n, 45, 90)  # 调整形状为 (batch, n, 45, 90)

        x = x_middle

        # # 上采样部分
        # # 跳跃连接 1：skip3 -> 上采样后大小对齐
        # skip3_flat = skip3.view(skip3.size(0), -1)  # 展平 skip3
        # skip3_proj = self.fc_skip3(skip3_flat).view(x_middle.size())  # 映射到 (batch, n, 45, 90)
        # x = x_middle + skip3_proj  # 跳跃连接
        
        x = self.upconv1(x)  # (batch, n*4, 90, 180)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        
        # # 跳跃连接 2：skip2 -> 上采样后大小对齐
        # skip2_flat = skip2.view(skip2.size(0), -1)  # 展平 skip2
        # skip2_proj = self.fc_skip2(skip2_flat).view(x.size(0), n * 4, 90, 180)  # 映射到 (batch, n*4, 90, 180)
        # x = x + skip2_proj  # 跳跃连接
        
        x = self.upconv2(x)  # (batch, n*2, 180, 360)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        
        # # 跳跃连接 3：skip1 -> 上采样后大小对齐
        # skip1_flat = skip1.view(skip1.size(0), -1)  # 展平 skip1
        # skip1_proj = self.fc_skip1(skip1_flat).view(x.size(0), n * 2, 180, 360)  # 映射到 (batch, n*2, 180, 360)
        # x = x + skip1_proj  # 跳跃连接
        
        x = self.upconv3(x)  # (batch, 1, 360, 720)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        
        # 最终 1x1 卷积
        x = self.conv1x1(x)
        
        return x

decoder = Decoder(n=4)
x = torch.rand(5, 22500, 576)  # 随机生成输入张量，范围 [0, 1]
y = decoder(x,n=4)
print(y.shape)
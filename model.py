import torch
from torch import nn
import torchvision
from torchvision.datasets import Cityscapes
from torch.utils.data.sampler import SubsetRandomSampler
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='small', is_pretrain = True,freeze=False, load_from=None):
        super().__init__()

        BACKBONE_SIZE = version # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', backbone_name)
        self.dinov2 = torch.hub.load('DINOv2', backbone_name, source='local', pretrained=is_pretrain)
        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)

        self.freeze = freeze

    def forward(self, inputs):
        B, _, h, w = inputs.shape

        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 4)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 4)

        outs = []
        for i,feature in enumerate(features):
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)

        return outs

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes,target_shape = (1024,2048)):
        super(SegmentationHead, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Upsample(target_shape, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.seg_head(x)

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        # print(f"Before pooling: {x.shape}")
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts

class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6],num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        # print(f" out: {out.shape}")
        out = self.final(out)
        return out

class FPNHEAD(nn.Module):
    def __init__(self, channels=384, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),  # 最终输出 num_classes 通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)
    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2)*2, x1.size(3)*2), mode='bilinear', align_corners=True)
        x_ = nn.functional.interpolate(input_fpn[-2], size=(x1.size(2)*2, x1.size(3)*2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(x_)
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2)*2, x2.size(3)*2), mode='bilinear', align_corners=True)
        x_ = nn.functional.interpolate(input_fpn[-3], size=(x2.size(2)*2, x2.size(3)*2), mode='bilinear', align_corners=True)

        x = x + self.Conv_fuse2(x_)

        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2)*2, x3.size(3)*2), mode='bilinear', align_corners=True)
        x_ = nn.functional.interpolate(input_fpn[-4], size=(x3.size(2)*2, x3.size(3)*2), mode='bilinear', align_corners=True)

        x = x + self.Conv_fuse3(x_)
        x4 = self.Conv_fuse3_(x)


        x1 = nn.functional.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = nn.functional.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = nn.functional.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, version='small', num_classes=21, pretrained=True):
        super(DINOv2SegmentationModel, self).__init__()
        self.backbone = DINOv2(version=version, freeze=True, is_pretrain=True)
        self.FPN = FPNHEAD(channels=384, out_channels=256)  # 替换为 FPNHEAD
        self.segmentation_head = SegmentationHead(in_channels=256, num_classes=num_classes,target_shape = (224,224))  # 1024 可根据 DINOv2 输出通道数调整

    def forward(self, x):
        # 获取多尺度特征
        features = self.backbone(x)
        # 将多尺度特征传递给分割头
        FPN_feature = self.FPN(features)

        segmentation_output = self.segmentation_head(FPN_feature)

        return segmentation_output

# 通用评估函数
def evaluate(model, loader, criterion, device, phase="validation", num_classes=20,save_path  = None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions, targets_list, images_list = [], [], []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"{phase.capitalize()} Phase", leave=False, dynamic_ncols=True)
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.squeeze(1).to(device)
            targets = (targets*255).long()

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            # 保存推理结果、标签和原始图像
            predictions.append(predicted.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            images_list.append(images.cpu())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    print(f"{phase.capitalize()} Loss: {avg_loss:.4f}, {phase.capitalize()} Accuracy: {accuracy:.2f}%")

    # 可视化并保存
    if save_path:
        save_path_full = f"{save_path}/{phase}_visualization.png"
        visualize_predictions(images_list[0], predictions[0], targets_list[0], phase, save_path=save_path_full)
    else:
        visualize_predictions(images_list[0], predictions[0], targets_list[0], phase)

    return avg_loss, accuracy

cityscapes_colors = [
    (128, 64, 128),   # 道路
    (244, 35, 232),   # 人行道
    (70, 70, 70),     # 建筑物
    (102, 102, 156),  # 墙壁
    (190, 153, 153),  # 栅栏
    (153, 153, 153),  # 灯杆
    (250, 170, 30),   # 信号灯
    (220, 220, 0),    # 标志牌
    (107, 142, 35),   # 树木
    (152, 251, 152),  # 地形
    (70, 130, 180),   # 天空
    (220, 20, 60),    # 人
    (255, 0, 0),      # 骑行者
    (0, 0, 142),      # 汽车
    (0, 0, 70),       # 卡车
    (0, 60, 100),     # 公交车
    (0, 80, 100),     # 火车
    (0, 0, 230),      # 摩托车
    (119, 11, 32),    # 自行车
]

def map_to_color(label_map, color_map):
    height, width = label_map.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in enumerate(color_map):
        color_image[label_map == label] = color
    return color_image

# 定义 visualize_predictions 函数
def visualize_predictions(images, predictions, targets_list, phase, save_path = None):
    # 设置可视化的图像数量
    num_images = min(4, len(images))  # 确保不会超出图像总数
    fig, axes = plt.subplots(num_images, 4, figsize=(15, num_images * 3))

    for i in range(num_images):
        img = images[i].permute(1, 2, 0)  # 将通道从 (C, H, W) 转为 (H, W, C)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # 去标准化
        img = img.numpy().clip(0, 1)

        # 绘制原始图像
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{phase.capitalize()} Image {i+1}")
        axes[i, 0].axis("off")

        # 绘制预测分割图（直接显示类别索引）
        prediction = map_to_color(predictions[i], cityscapes_colors)
        axes[i, 1].imshow(prediction)
        axes[i, 1].set_title(f"{phase.capitalize()} Prediction {i+1}")
        axes[i, 1].axis("off")

        # 绘制真实分割图（直接显示类别索引）
        target = map_to_color(targets_list[i], cityscapes_colors)
        axes[i, 2].imshow(target)
        axes[i, 2].set_title(f"{phase.capitalize()} Target {i+1}")
        axes[i, 2].axis("off")

        # 叠加图像
        # print(targets_list.shape)
        overlay = (0.5 * img + 0.5 * (prediction / prediction.max())).clip(0, 1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"{phase.capitalize()} Overlay {i+1}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    # 保存图像到文件（如果提供路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
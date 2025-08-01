import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

class StyleEvaluator:
    def __init__(self):
        # 加载预训练VGG19（用于提取风格特征）
        self.vgg = models.vgg19(pretrained=True).features.eval()

        # # 创建VGG19模型结构（不加载预训练权重）
        # vgg = models.vgg19(pretrained=False)  # pretrained=False表示不自动下载权重
        # # 从本地加载权重
        # local_weights_path = 'vgg19_pretrained.pth'  # 替换为你的本地权重文件路径
        # vgg.load_state_dict(torch.load(local_weights_path))
        # # 只使用特征提取部分（前几层卷积和池化）
        # vgg_features = vgg.features.eval()  # 设置为评估模式

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 风格特征提取层（选取relu1_1到relu3_1的特征）
        self.style_layers = [0, 2, 5, 7, 10]  # VGG19的卷积层索引

    def _extract_style_features(self, image_path):
        """提取图像的风格特征（Gram矩阵的均值）"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # 增加批次维度
        
        features = []
        x = img_tensor
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.style_layers:
                # 计算Gram矩阵（风格特征的核心）
                b, c, h, w = x.shape
                gram = torch.matmul(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1, 2))
                gram = gram / (c * h * w)  # 归一化
                features.append(gram.mean().item())  # 简化为均值，实际可保留完整向量
        return np.array(features)

    def compute_score(self, generated_image_path, target_style_image_path):
        """
        计算风格一致性得分
        :param generated_image_path: 生成图像路径
        :param target_style_image_path: 目标风格图像路径（如梵高画作）
        :return: style_score (0~1，余弦相似度归一化后)
        """
        # 提取生成图像和目标风格图像的特征
        gen_feats = self._extract_style_features(generated_image_path)
        target_feats = self._extract_style_features(target_style_image_path)
        
        # 计算余弦相似度（越大越相似）
        cos_sim = 1 - cosine(gen_feats, target_feats)
        # 归一化到[0,1]（余弦相似度范围为[-1,1]）
        style_score = (cos_sim + 1) / 2
        return round(style_score, 3)
#!/usr/bin/env python
# coding=utf-8
import torch


def pairwise_metrics(x, metric='cosine', reduction=None):
    """
    计算形状为 [b, n, h] 的张量每个 batch 样例中 n 个回复之间的两两度量值。
    返回形状为 [b, s] 的张量, s = n * (n - 1) / 2。
    
    :param x: 输入张量, 形状为 [b, n, h]
    :param metric: 度量方式, 可选 'cosine', 'euclidean', 'pearson'
    :param reduction: 简化操作, 可选 None 或 'mean'
    :return: 度量值张量, 形状为 [b, s] 或 [b] (如果 reduction='mean')
    """
    b, n, h = x.shape
    # 计算上三角矩阵的索引（排除对角线）
    indices = torch.triu_indices(n, n, 1, device=x.device)
    
    # 提取出所有需要计算的回复对
    x_i = x[:, indices[0]]  # [b, s, h]
    x_j = x[:, indices[1]]  # [b, s, h]
    
    if metric == 'cosine':
        # 余弦相似度计算
        norm_i = torch.norm(x_i, dim=-1, keepdim=True)  # 计算向量的范数
        norm_j = torch.norm(x_j, dim=-1, keepdim=True)
        cosine_similarity = torch.sum(x_i * x_j, dim=-1) / (norm_i * norm_j + 1e-8)
        result = cosine_similarity
    
    elif metric == 'euclidean':
        # 欧式距离计算
        euclidean_distance = torch.norm(x_i - x_j, dim=-1)
        result = euclidean_distance
    
    elif metric == 'pearson':
        # 皮尔逊系数计算
        mean_i = torch.mean(x_i, dim=-1, keepdim=True)
        mean_j = torch.mean(x_j, dim=-1, keepdim=True)
        
        x_i_centered = x_i - mean_i  # 去中心化
        x_j_centered = x_j - mean_j
        
        numerator = torch.sum(x_i_centered * x_j_centered, dim=-1)
        denominator = torch.sqrt(torch.sum(x_i_centered ** 2, dim=-1) * torch.sum(x_j_centered ** 2, dim=-1) + 1e-8)
        pearson_correlation = numerator / denominator
        result = pearson_correlation

    else:
        raise ValueError(f"不支持的度量方式: {metric}")
    
    if reduction == 'mean':
        result = torch.mean(result, dim=-1)  # 在 s 维度上取均值, 得到形状为 [b] 的张量
    
    return result
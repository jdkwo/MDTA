"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
from scipy import spatial
import torch
import torch.nn as nn
from networks.resnet_big import Discriminator


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        for l in range(1, labels.shape[1]):
            mask[:, labels.shape[1] - l:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / (l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        return cumulative_loss / labels.shape[1]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [batch, dim].
            labels: ground truth of shape [batch].
            mask: contrastive mask of shape [batch, batch], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = 1

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # 所有正样本的点乘运算
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, 1)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + 1e-9)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数

    def forward(self, inputs, labels):
        n = inputs.size(0)  # 获取batch_size，这里的inputs就是输入矩阵,即batchsize * 特征维度
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2, )  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里mask[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss



class DomainLoss(nn.Module):
    def __init__(self, out_dim):
        super(DomainLoss, self).__init__()
        self.Discr = nn.ModuleDict({
            '0': Discriminator(out_num=out_dim),
            '1': Discriminator(out_num=out_dim),
            '2': Discriminator(out_num=out_dim),
            '3': Discriminator(out_num=out_dim),
            '4': Discriminator(out_num=out_dim),
            '5': Discriminator(out_num=out_dim)
        })
        self.cr = torch.nn.CrossEntropyLoss()

    def forward(self, feature, labels, positions):
        loss = 0
        for i in range(0, 6):
            for j in range(0, len(labels)):
                if labels[j] == i:
                    feat = self.Discr[str(i)](feature[j, :])
                    loss = loss + self.cr(feat, positions[j])
        return loss / len(labels)

"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import inception
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        dim_in = 2000
        self.encoder = inception.InceptionTime()

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name='resnet50', num_classes=6):
        super(SupCEResNet, self).__init__()
        self.encoder = inception.InceptionTime()

    def forward(self, x):
        feat = self.encoder(x)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', num_classes=7):
        super(LinearClassifier, self).__init__()
        feat_dim = 1024
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class Discriminator(nn.Module):

    def __init__(self, out_num):
        super(Discriminator, self).__init__()
        feat_dim = 1024
        self.layer1 = nn.Linear(feat_dim, 128)
        self.layer2 = nn.Linear(128, out_num)
        self.relu = nn.ReLU()

    def forward(self, features):
        out = self.layer1(features)
        out = self.relu(out)
        out = self.layer2(out)
        return out


class CustomGNNLayer(MessagePassing):
    def __init__(self):
        super(CustomGNNLayer, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # Step 3: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j has shape [num_edges, out_channels]
        # edge_weight has shape [num_edges]

        # Step 4: Normalize node features by their edge weight
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]
        return F.relu(aggr_out)


class Disentanglement(nn.Module):
    def __init__(self):
        super(Disentanglement, self).__init__()
        self.fc1 = nn.Linear(1200, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 5120)
        self.bn3 = nn.BatchNorm1d(5120)


    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x_1 = x[:, 0:1024]
        x_2 = x[:, 1024:2048]
        x_3 = x[:, 2048:3072]
        x_4 = x[:, 3072:4096]
        x_5 = x[:, 4096:5120]
        return x_1, x_2, x_3, x_4, x_5


class recon_net(nn.Module):
    def __init__(self):
        super(recon_net, self).__init__()
        self.fc1 = nn.Linear(5120, 2048)
        self.fc2 = nn.Linear(2048, 1200)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.data import Data, Batch
from sklearn.metrics.pairwise import cosine_similarity


# import faiss

class MomentumQueueClass(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes, eps_ball=1.1):
        super(MomentumQueueClass, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer('memory_label', memory_label)

    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def extend_test(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            self.memory = torch.cat([self.memory, k_all], dim=0)
            self.memory_label = torch.cat([self.memory_label, k_label_all], dim=0)

    def forward(self, x, test=False):
        dist = torch.mm(F.normalize(x), self.memory.transpose(1, 0))  # B * Q, memory already normalized
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_weight, sim_indices = torch.topk(dist, k=self.k)
        sim_labels = torch.gather(self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices)
        # sim_weight = (sim_weight / self.temperature).exp()
        if not test:
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
        else:
            sim_weight = F.softmax(sim_weight / 0.1, dim=1)

        # counts for each class
        one_hot_label = torch.zeros(x.size(0) * self.k, self.classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, self.classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores


class MomentumQueue(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes, eps_ball=1.1):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes
        self.eps_ball = eps_ball

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer('memory_label', memory_label)

    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def extend_test(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            self.memory = torch.cat([self.memory, k_all], dim=0)
            self.memory_label = torch.cat([self.memory_label, k_label_all], dim=0)

    def reduce_test(self):
        with torch.no_grad():
            long = len(self.memory_label)
            self.memory = torch.split(self.memory, [self.queue_size, long-self.queue_size], dim=0)[0]
            self.memory_label = torch.split(self.memory_label, [self.queue_size, long-self.queue_size], dim=0)[0]

    def forward(self, x, model, num_cls):
        samples = torch.cat([self.memory, x], dim=0)
        data = self.build_graph(samples, num_cls)
        output = model(data)
        data.x = F.softmax(output, dim=1)
        output = model(data)
        data.x = F.softmax(output, dim=1)
        output = model(data)
        probabilities = F.softmax(output, dim=1)
        confidence, predict = probabilities.max(dim=1)
        return confidence[-x.size(0):], predict[-x.size(0):]


    def build_graph(self, x, num_cls):
        """
        构建一个 K-近邻图，其中边的权重由节点之间的余弦相似度决定。
        x: 输入的节点特征矩阵, 大小为 [num_nodes, feature_dim]
        K: 每个节点的 K 个最近邻
        """
        # 计算余弦相似度矩阵
        k = self.k
        x = x.cpu()
        cos_sim_matrix = cosine_similarity(x.numpy())  # 计算节点特征之间的余弦相似度

        # 创建边的索引和权重
        edge_index = []
        edge_weight = []

        # 为每个节点找到 K 个最相似的邻居
        for i in range(x.size(0)):
            # 获取该节点与所有其他节点之间的相似度
            similarity = cos_sim_matrix[i]

            # 将相似度按降序排序并选择前 K 个最近邻（排除自己）
            sorted_indices = np.argsort(similarity)[::-1][1:k + 1]  # 排除自己，从第 1 个开始
            sorted_similarities = similarity[sorted_indices]

            # 构建边和对应的权重
            for idx, sim in zip(sorted_indices, sorted_similarities):
                #edge_index.append([i, idx])  # i 节点和 idx 节点之间有边
                #edge_weight.append(sim)  # 边的权重是相似度
                # 添加反向边
                edge_index.append([idx, i])  # idx 节点和 i 节点之间有边
                edge_weight.append(sim)  # 反向边的权重也是相似度

        # 转换为 PyTorch Geometric 格式
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)  # 边的权重是余弦相似度

        # 创建图数据对象
        batch_size = x.size(0)-self.memory_label.size(0)
        x = F.one_hot(self.memory_label, num_cls).to(torch.float32)
        x = torch.cat([x, torch.zeros((batch_size, num_cls)).cuda()], dim = 0)
        data = Data(x, edge_index=edge_index.cuda(), edge_attr=edge_weight.cuda())
        return data


    def test_graph(self, feat, label, num_cls):
        batch_graphs = []
        batch_size = feat.size(0)
        for i in range(batch_size):
            similarity = F.cosine_similarity(feat[i].unsqueeze(0), self.memory, dim=1)
            topk_indices = similarity.topk(self.k, largest=True).indices

            # 创建图结构
            node_features = torch.cat([label[i].unsqueeze(0), self.memory_label[topk_indices]], dim=0)
            edge_index = [[0] * self.k + list(range(1, self.k + 1)), list(range(1, self.k + 1)) + [0] * self.k]
            edge_weight = similarity[topk_indices].repeat(2)  # 添加相似度权重

            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            data = Data(x=F.one_hot(node_features, num_cls).to(torch.float32), edge_index=edge_index,
                        edge_attr=edge_weight)

            batch_graphs.append(data)

        return Batch.from_data_list(batch_graphs)




class Model_with_Predictor(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, args, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Model_with_Predictor, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        if args.mlp:
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # second layer
                                            self.encoder.fc,
                                            nn.BatchNorm1d(dim, affine=False))  # output layer
            self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return p, z


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.module.encoder.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

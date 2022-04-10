import math
# from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(torch.nn.Module):
    def __init__(self, feature_dim, dim_latent, is_pruning=True):
        super(TeacherModel, self).__init__()
        self.feature_dim = feature_dim
        self.dim_latent = dim_latent
        self.decay_ration = 2
        self.is_pruning = is_pruning
        # 教师网络通过多层进行特征变换
        liner_feat_dim = self.feature_dim
        linear_list = []
        while True:
            liner_feat_dim = int(liner_feat_dim / self.decay_ration)
            # print(liner_feat_dim)
            if liner_feat_dim < self.dim_latent:
                linear_list.append(nn.Linear(liner_feat_dim * self.decay_ration, self.dim_latent))
                break
            else:
                linear_list.append(nn.Linear(liner_feat_dim * self.decay_ration, liner_feat_dim))
                if liner_feat_dim == self.dim_latent:
                    break
                # linear_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=4))
        self.transfer_multilayer = nn.Sequential(*linear_list)

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.unsqueeze(1)
        transfer_x = self.transfer_multilayer(x)
        # transfer_x = transfer_x.view(batch_size, self.dim_latent)
        if self.is_pruning:
            transfer_x = F.leaky_relu(transfer_x)

        return transfer_x


class StudentModel(torch.nn.Module):
    def __init__(self, feature_dim, dim_latent, is_pruning=True):
        super(StudentModel, self).__init__()
        self.dim_latent = dim_latent
        self.feature_dim = feature_dim
        self.is_pruning = is_pruning
        # 线性变换器
        # todo:可以尝试多种trick方式
        self.MLP = nn.Linear(self.feature_dim, self.dim_latent)
        # self.transfer_layer = nn.Linear(self.dim_latent, self.dim_latent)

    def forward(self, x):
        transfer_x = self.MLP(x)
        # transfer_x = self.transfer_layer(x)
        if self.is_pruning:
            transfer_x = F.leaky_relu(transfer_x)

        return transfer_x

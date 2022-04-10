import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_fusion.attention import MultiheadedAttention

class TeacherModel(torch.nn.Module):
    def __init__(self, meta_label, dim_latent):
        super(TeacherModel, self).__init__()
        self.dim_latent = dim_latent
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))
        self.feature_transfer_layer = nn.Linear(2 * dim_latent, dim_latent)
        # self.feature_transfer_layer = nn.Sequential(
        #     nn.Linear(2 * dim_latent, dim_latent),
        #     # nn.Linear(2 * dim_latent, dim_latent),
        # )
        self.category_classification = nn.Linear(self.dim_latent, self.label_size)

    def forward(self, nodes, fusion, has_n=True):

        gt_prediction = self.meta_label[nodes]

        transfer_layer = self.feature_transfer_layer(fusion)

        if has_n:
            transfer_layer = F.leaky_relu_(transfer_layer)

        result = self.category_classification(transfer_layer)

        class_loss = F.cross_entropy(result, gt_prediction)

        return class_loss


class StudentModel(torch.nn.Module):
    def __init__(self, dim_latent=64):
        super(StudentModel, self).__init__()
        self.dim_latent = dim_latent
        # 用来变换特征到相同的空间
        self.v_feat_linear = nn.Linear(dim_latent, dim_latent)
        self.t_feat_linear = nn.Linear(dim_latent, dim_latent)
        # 特征融合层
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2 * dim_latent, 2),
            nn.Softmax(dim=-1),
        )
        self.LayerNorm = nn.LayerNorm(dim_latent)

        # self.feature_fusion = nn.Linear(2 * self.dim_latent, 2 * self.dim_latent)

    def forward(self, v_compacted_feat, t_compacted_feat):

        transferred_v_feat = self.v_feat_linear(v_compacted_feat)
        transferred_t_feat = self.v_feat_linear(t_compacted_feat)

        fused_feature = torch.cat([transferred_v_feat, transferred_t_feat], dim=-1)

        # feat_embeds = [transferred_v_feat, transferred_t_feat]
        # # 针对不同的特征首先进行特征变换
        # union_feature = torch.cat(feat_embeds, dim=-1)
        # attention_scores = self.attention(union_feature)
        #
        # fused_feature = attention_scores.unsqueeze(-1) * torch.stack(feat_embeds, dim=1)
        # # print(fused_feature.shape)
        # fused_feature = fused_feature.sum(dim=1)
        #
        # fused_feature = self.LayerNorm(fused_feature)

        return fused_feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor.TeacherStudentModel import TeacherModel,StudentModel


class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, feature, feature_dim, meta_label,dim_latent=64, is_pruning=True):
        super(FeatureExtractorModel, self).__init__()
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))
        self.dim_latent = dim_latent
        self.is_pruning = is_pruning
        # 初始化item的特征
        # 输入的item特征不参与训练
        self.features = feature
        # 温度系数，用于软化目标结果
        self.t = 2
        self.a = 0.7
        # print(self.dim_latent, self.label_size)
        # MLP分类器
        self.category_classification = nn.Linear(self.dim_latent, self.label_size)

        # self.student_classification = nn.Linear(self.dim_latent, self.label_size)

        self.teacher_model = TeacherModel(feature_dim, self.dim_latent,  self.is_pruning)
        self.student_model = StudentModel(feature_dim, self.dim_latent,  self.is_pruning)

    def forward(self, nodes):
        node_feature = self.features[nodes]
        node_label = self.meta_label[nodes]
        # 1. 首选获得教师网络的分类结果
        teacher_x = self.teacher_model(node_feature)
        # print(node_feature.shape, teacher_x.shape)
        teacher_result = self.category_classification(teacher_x)
        teacher_soft_result = F.softmax(teacher_result / self.t, dim=-1)
        teacher_hard_result = F.softmax(teacher_result, dim=-1)

        # 2. 获得学生网络的分类结果
        student_x = self.student_model(node_feature)
        # print(student_x.shape)
        student_result = self.category_classification(student_x)
        student_soft_result = F.softmax(student_result / self.t, dim=-1)
        # student_hard_result = F.softmax(student_result, dim=-1)

        # 3. 获得教师的分类loss
        extra_teacher_loss = F.cross_entropy(teacher_hard_result, node_label)
        # extra_student_loss = F.cross_entropy(student_hard_result, node_label)

        # 4. 计算教师网络和学生网络的知识蒸馏loss
        extra_class_KD_loss = F.l1_loss(teacher_soft_result, student_soft_result)
        extra_feature_KD_loss = F.mse_loss(student_x, teacher_x)
        extra_KD_loss = 0.5 * extra_feature_KD_loss + extra_class_KD_loss
        # print(extra_feature_KD_loss, extra_class_KD_loss)
        return teacher_x, extra_teacher_loss, extra_KD_loss

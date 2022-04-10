import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_fusion.TeacherStudentModel import TeacherModel, StudentModel


class FeatureFusionModel(torch.nn.Module):
    def __init__(self, image_feature, text_feature, meta_label, dim_latent=64):
        super(FeatureFusionModel, self).__init__()
        self.image_feature = image_feature
        self.text_feature = text_feature
        self.meta_label = meta_label
        self.label_size = len(set(meta_label.tolist()))
        self.dim_latent = dim_latent
        self.t = 1
        self.teacher_model = TeacherModel(self.image_feature, self.text_feature, self.dim_latent)
        self.student_model = StudentModel(self.dim_latent)
        # print(self.label_size)
        # MLP分类器
        self.category_classification = nn.Linear(2 * self.dim_latent, self.label_size)

    def forward(self, nodes, image_compacted_feature, text_compacted_feature):
        gt_prediction = self.meta_label[nodes]
        # 处理教师网络的结果
        teacher_feature = self.teacher_model(nodes)
        teacher_result = self.category_classification(teacher_feature)
        teacher_soft_result = F.softmax(teacher_result / self.t, dim=-1)
        teacher_hard_result = F.softmax(teacher_result, dim=-1)

        teacher_class_loss =  F.cross_entropy(teacher_hard_result, gt_prediction)

        # 处理学生网络的结果
        student_feature = self.student_model(image_compacted_feature, text_compacted_feature)
        student_result = self.category_classification(student_feature)
        student_soft_result = F.softmax(student_result / self.t, dim=-1)
        student_hard_result = F.softmax(student_result, dim=-1)

        # student_class_loss = F.cross_entropy(student_hard_result, gt_prediction)
        # mse_loss = F.mse_loss(teacher_soft_result, student_soft_result)
        feature_mse_loss = F.mse_loss(teacher_feature, student_feature)

        # 分别计算loss
        # 通过计算教师网络和学生网络的软目标损失，使学生网络学习到更多的教师网络的语义信息

        return student_feature, teacher_class_loss, 10 * feature_mse_loss


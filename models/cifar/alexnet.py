'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import copy
import torch
from .selayer import SELayer

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=31):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.selayer3 = SELayer(384)
        self.selayer4 = SELayer(256)
        self.selayer5 = SELayer(256)
        self.ad_bn_3 = nn.AD_BatchNorm2d(384)
        self.ad_bn_4 = nn.AD_BatchNorm2d(256)
        self.ad_bn_5 = nn.AD_BatchNorm2d(256)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            #SELayer(64)
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.InstanceNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        )
        #    nn.BatchNorm2d(384),
        #    nn.ReLU(inplace=True),
        #)
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
        )
        #    nn.BatchNorm2d(256),
        #    nn.ReLU(inplace=True),
        #)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
        )
       #     nn.BatchNorm2d(256),
       #     nn.ReLU(inplace=True),
       #     nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(256),
        #)
        
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.InstanceNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.copy_feature = copy.deepcopy(self.features)
        #print(self.features.1)
        self.bn = nn.BatchNorm1d(256)
        self.ad_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x_copy = x
        
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x3_weight = self.selayer3(x3)
        x3 = self.relu(self.ad_bn_3(x3, x3_weight))
        
        x4 = self.block4(x3)
        x4_weight = self.selayer4(x4)
        x4 = self.relu(self.ad_bn_4(x4, x4_weight))
        
        x = self.block5(x4)
        x5_weight = self.selayer5(x)
        x = self.maxpool(self.relu(self.ad_bn_5(x, x5_weight)))
        
        x = self.ad_pool(x)
        x_kl = x.view(x.size(0), -1)
        x_kl = self.bn(x_kl)
        x = self.classifier(x_kl)
        return x, x_kl, x1, x2


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model

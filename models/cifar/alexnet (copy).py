'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import copy
import torch

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.copy_feature = copy.deepcopy(self.features)
        self.classifier = nn.Linear(65536, num_classes)

    def forward(self, x):
        x_copy = x
        x = self.features(x) # x--> [128 256 1 1]
        x_copy = self.copy_feature(x_copy)
        x = x.view(x.size(0), x.size(1), -1) # [128 256 1]
        x_copy = x_copy.view(x_copy.size(0), x_copy.size(1), -1)
        #print(x.shape)
        final_x = torch.bmm(x, x_copy.permute(0, 2, 1)) # [128 256 256]
        final_x = final_x.view(x.size(0), -1) # [128 256*256]
        #x = x.view(x.size(0), -1)
        x = self.classifier(final_x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model

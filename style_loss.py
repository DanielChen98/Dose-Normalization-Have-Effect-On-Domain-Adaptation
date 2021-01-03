import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class style_loss(nn.Module):

    def __init__(self, x):
        super(style_loss, self).__init__()
        self.target = gram_matrix(x).detach()

    def forward(self, y):
        G = gram_matrix(y)
        loss = F.mse_loss(G, self.target)
        return loss

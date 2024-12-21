import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptcnn_utils.model import RandPointCNN
from ptcnn_utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from ptcnn_utils.util_layers import Dense


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, num_class, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = x.transpose(1,2) #back in shape batch_size*N*3
        x = self.pcnn1((x, x))

        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        # logits_mean = F.log_softmax(logits_mean, dim=1)
        return logits_mean, None


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)
        total_loss = F.cross_entropy(pred, target)

        return total_loss


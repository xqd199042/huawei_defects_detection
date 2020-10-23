from torch import nn


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

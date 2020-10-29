import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        # focal loss -alpha(1-yi)**gamma*ce_loss(yi,y*)
        self.size_average = size_average
        self.gamma = gamma
        if isinstance(alpha, list):
            assert len(alpha)==num_classes
            self.alpha = Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += 1-alpha

    def forward(self, logits, targets):
        predicts = softmax(logits.view(-1, logits.size(-1)), dim=1)
        self.alpha = self.alpha.to(logits.device)
        
























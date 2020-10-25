import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import softmax


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes

    def forward(self, x, targets):
        num_in_batch = x.size(0)
        num_logits = x.size(1)
        props = softmax(x)

        class_mask = Variable(x.data.new(num_in_batch, num_logits).fill_(0))
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if x.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        log_p = (props*class_mask).sum(1).view(-1, 1).log()
        batch_loss = -alpha*(torch.pow((1-props), self.gamma))*log_p

        loss = batch_loss.mean() if self.size_average else batch_loss.sum()
        return loss

























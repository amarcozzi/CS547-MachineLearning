from numpy import dtype
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy, cross_entropy

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=False):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, device, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device, dtype=torch.float32)
        self.gamma = torch.tensor(gamma).to(device, dtype=torch.float32)

    def forward(self, inputs, targets):
        loss = cross_entropy(inputs, targets, reduction='none')
        at = self.alpha.gather(0, targets.data.view(-1)).reshape(targets.shape)
        pt = torch.exp(-loss)
        F_loss = at*(1-pt)**self.gamma * loss
        return F_loss.mean()
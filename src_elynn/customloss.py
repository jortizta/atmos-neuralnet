import torch.nn as nn
import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F

class customLoss(_WeightedLoss):

    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction='none'):
        super(customLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

#    def forward(self, input, target):
 #       return F.cross_entropy(input, target, weight=self.weight,
  #              ignore_index=self.ignore_index, reduction=self.reduction)



    def forward(self, input, target):
        loss = F.l1_loss(input, target, reduction=self.reduction)
        if (self.ignore_index >= 0):
            loss = loss[:,:,0:self.ignore_index,:]
            if not (type(self.weight) == type(None)):
                self.weight = self.weight[:,:,0:self.ignore_index,:]

        if not (type(self.weight) == type(None)):
            #print("help")
            #print(loss.shape, self.weight.shape)
            loss = loss*self.weight
        #print(loss.shape)
        #print(loss.sum()/(loss.shape[0]*loss.shape[1]*loss.shape[2]*loss.shape[3]))
        #print("\n\n")
        return loss.sum()/(loss.shape[0]*loss.shape[1]*loss.shape[2]*loss.shape[3])


import torch 
from torch import nn 
from torch.nn import functional as F 

class CrossEntropyLoss(nn.Module): 
    def __init__(self): 
        super(CrossEntropyLoss, self).__init__() 
    
    def forward(self, x, target): 
        return torch.mean(torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)) 
    
class MatchingLoss(nn.Module): 
    def __init__(self): 
        super(MatchingLoss, self).__init__() 
        
    def forward(self, x, target): 
        return F.mse_loss(torch.softmax(x, dim=-1), target)
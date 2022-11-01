
import torch
from torch import nn
import pdb

class LinearPredExtended(nn.Module):
    def __init__(self, nevents=5):
        super().__init__()
        self.nevents = nevents
        self.betas = nn.parameter.Parameter(torch.rand(nevents+1,nevents))

    def forward(self, input):
        device = input.device
        conds = input[:,1:]
        events = nn.functional.one_hot(input[:,0].long(), num_classes=self.nevents)
        betas = (self.betas.to(device) @ events.transpose(0,1).float()).transpose(0,1)
        X = torch.stack([(conds==cnd).any(dim=1).int() for cnd in range(self.nevents)]).transpose(0,1).float()                
        Xc = torch.concat((X, torch.ones((X.shape[0], 1), device=device)), dim=1)
        z = (Xc * betas).sum(dim=1)
        return torch.sigmoid(z)


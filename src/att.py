
import torch
from torch import nn
import pdb

class CondFilterT(nn.Module):
    
    def __init__(self, nevents=5, nconds=1, emb_dim=5, nparams=1):
        super().__init__()
        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)
        self.last_loss = None
        #self.nparams = nparams
        #self.output_size = emb_dim * 2

    def forward(self, input):
        event = input[:,0:1]
        conditions = input[:,1:]

        event_emb = self.event_embedding(event)
        cond_emb = self.event_embedding(conditions)

        event_emb_nrm = event_emb / event_emb.norm(dim=2, keepdim=True)
        cond_emb_nrm = cond_emb / cond_emb.norm(dim=2, keepdim=True)

        scores = event_emb_nrm @ cond_emb_nrm.transpose(1,2)
        self.last_loss = scores.abs().sum()
        print("###", self.last_loss)
        filtered_cond = cond_emb_nrm * scores.transpose(1,2)
        return torch.concat((event_emb.flatten(start_dim=1), filtered_cond.flatten(start_dim=1)), dim=1)
        #norm_scores = torch.softmax(scores)        
        #return scores


class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
        #self.ff = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU()])##, nn.BatchNorm1d(dim)])

    def forward(self, input):
        return self.ff(input) + input


class SimplePred(nn.Module):
    def __init__(self, nevents=5, emb_dim=20, nconds=10, depth=4):
        super().__init__()

        self.nconds = nconds
        self.emb_dim = emb_dim

        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        inner_dim = emb_dim * 2
        self.inner_dim =inner_dim

        self.chain = nn.Sequential(*
            ([Residual(inner_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):
        cond_emb = self.cond_embedding(input[:,1:].int())
        #if self.training:
        #    cond_emb += (torch.rand(cond_emb.shape, device=input.device) - 0.5) / 1000
        bag = cond_emb.sum(dim=1) 

        #bnorm =  bag.norm(dim=1, keepdim=True)   
        #bnorm = torch.where(bnorm > 0, bnorm, torch.ones(bnorm.shape, device=input.device))
        #bag = bag / bnorm
        event_emb = self.event_embedding(input[:, 0].int())
        base = torch.concat((event_emb, bag), dim=1)
        return self.chain(base)


class SimplePred2(nn.Module):
    def __init__(self, nevents=5, emb_dim=20, nconds=10, depth=4):
        super().__init__()

        self.nconds = nconds
        self.emb_dim = emb_dim

        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        inner_dim = emb_dim * 2
        self.inner_dim =inner_dim

        self.chain = nn.Sequential(*
            ([Residual(inner_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):

        event = input[:,0:1].int()
        conditions = input[:,1:].int()

        event_emb = self.event_embedding(event)
        cond_emb = self.event_embedding(conditions)

        #event_emb_nrm = event_emb / event_emb.norm(dim=2, keepdim=True)
        #cond_emb_nrm = cond_emb / cond_emb.norm(dim=2, keepdim=True)

        scores = event_emb @ cond_emb.transpose(1,2)
        
        weighted_att = torch.softmax(scores, dim=2).transpose(1,2) * cond_emb
        att = weighted_att.sum(dim=1)
        base = torch.concat((event_emb.flatten(start_dim=1), att), dim=1)
        
        return self.chain(base)

class TransPred(nn.Module):
    def __init__(self, nevents=5, emb_dim=20, nconds=10, transformer_layers=6, depth=4, drop_rate=0.1):
        super().__init__()

        self.drop_rate = drop_rate
        self.nconds = nconds
        self.emb_dim = emb_dim

        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        inner_dim = emb_dim * 2
        self.inner_dim =inner_dim
        dim_feedforward = emb_dim
        self.transformer = nn.Sequential(*[nn.TransformerEncoderLayer(
            d_model = emb_dim, nhead=5, dim_feedforward=emb_dim,
            activation="relu"    
        ) for idx in range(transformer_layers)])

        self.chain = nn.Sequential(*
            ([Residual(inner_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):

        event = input[:,0:1].int()
        conditions = input[:,1:].int()

        #if self.training:
        #    conditions = conditions * (torch.rand(conditions.shape, device=input.device) > self.drop_rate).int()

        event_emb = self.event_embedding(event)
        cond_emb = self.cond_embedding(conditions)

        transformer_input = torch.concat((event_emb, cond_emb), dim=1)

        transformer_output = self.transformer(transformer_input)

        chain_input = torch.concat((
            transformer_output[:,0:1,:].sum(dim=1),
            transformer_output[:,1:,:].mean(dim=1)), dim=1)

        rv = self.chain(chain_input)
        return rv


class LinearPred(nn.Module):
    def __init__(self, nevents=5):
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(nevents,1, bias=True), nn.Sigmoid())
        self.nevents = nevents

    def forward(self, input):
        input = input[:,1:]
        X = torch.stack([(input==cond).any(dim=1).int() for cond in range(self.nevents)]).transpose(0,1).float()                
        return self.inner(X)

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

class CondFilter(nn.Module):

    def __init__(self, nevents=5, emb_dim=5, inner_dim=10, output_dim=10, depth=8):
        super().__init__()
        self.cond_embedding = nn.Embedding(nevents+2, emb_dim)
        self.event_embedding = nn.Embedding(nevents+2, emb_dim)

        self.inner = nn.Sequential(*
            ([nn.Linear(emb_dim * 2, inner_dim)] +
            [Residual(inner_dim) for idx in range(depth)] +
            [nn.Linear(inner_dim, output_dim)])
        )
        self.loss = None
        

    def forward(self, input, return_weight=False):
        event = input[:,0]
        conditions = input[:,1]
        event_emb = self.event_embedding(event.int())
        cond_emb = self.cond_embedding(conditions.int())
        emb = torch.concat((event_emb, cond_emb), dim=1)
        out = self.inner(emb)
        return out
        weight = torch.sigmoid(out[:,0:1])
        
        tail = out[:, 1:]

        #norm = tail.norm(dim=1, keepdim=True)
        #norm = torch.where(norm == 0, torch.ones(norm.shape, device=input.device), norm)
        #output = weight * tail  / norm
        #if return_weight:
        #    return weight
        #import pdb
        #pdb.set_trace()
        #if self.loss is None:
        #    self.loss = weight.mean()
        #else:
        #    self.loss += weight.mean()
        return tail


class EventPredictor(nn.Module):

    def __init__(self, nevents=5, emb_dim=5, inner_dim=30, filter_inner_dim=30, depth=5, filter_depth=5):
        super().__init__()
        self.inner_dim = inner_dim
        self.cond_filter = CondFilter(nevents=nevents, emb_dim=emb_dim, 
            inner_dim=filter_inner_dim, output_dim=inner_dim, depth=filter_depth)
        self.chain = nn.Sequential(*
            ([Residual(inner_dim + emb_dim)  for idx in range(depth)] +
            [nn.Linear(inner_dim + emb_dim, 1), nn.Sigmoid()])
        )

    def forward(self, input):
        base = torch.zeros((input.shape[0], self.inner_dim), device=input.device)        
        for idx in range(1,input.shape[1]):
            base += self.cond_filter(input[:,(0,idx)])
        event = input[:,0]
        event_emb = self.cond_filter.event_embedding(event.int())

        return self.chain(torch.concat((event_emb,base), dim=1))



#model = EventPredictor(nevents=12)
#input = torch.tensor([[5,2,3], [1,3,9], [1,4,5]])
#rv = model(input)
#print(rv)
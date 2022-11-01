
import torch
from torch import nn
import logging
import pdb

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


class Adaptor(nn.Module):
    
    def __init__(self, nevents=5, nconds=1, emb_dim=5, nparams=1):
        super().__init__()
        self.embedding = nn.Embedding(nevents+2, emb_dim)
        self.nparams = nparams
        self.output_size = emb_dim * 2

    def forward(self, input):
        item_size = 1 + self.nparams
        event_emb = self.embedding(input[:,0].int())
        cond_emb = self.embedding(input[:,item_size].int())
        return torch.concat((event_emb, cond_emb), dim=1)

class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU()])

    def forward(self, input):
        return self.ff(input) + input


def create_model(nevents=5, nconds=1, emb_dim=5, inter_dim=20, nparams=1, depth=8):
    adaptor = Adaptor(nevents=nevents, nconds=nconds, emb_dim=emb_dim, nparams=nparams)
    parts = (
        [adaptor, nn.Linear(adaptor.output_size, inter_dim)] +
        [Residual(inter_dim) for idx in range(depth)] +
        [nn.Linear(inter_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*parts)


def train_model(model, loss_fun, optimizer, sampler, repeats=100, batch_size=100, num_epochs=100, device=None, track=None):
    for idx in range(num_epochs):

        logging.info(f"{idx}")    
        ok = 0
        total =0 
        model.train()
        for rpt in range(repeats):
            data = sampler.sample(batch_size)
            if device is not None:
                data = data.to(device)
            input = data[:,0:-3]
            weight = data[:,-2]
            target = data[:,-1]
            optimizer.zero_grad()
            #model.cond_filter.loss = None
            y_model = model(input) 
            loss = (loss_fun(y_model.flatten(), target) * weight).sum() / weight.sum() ##+ 0.0 *  model.cond_filter.loss

            betas = model.betas[0:-1,:]
            beta_loss = (betas.abs()).sum() * 0.001            
            loss += beta_loss
            #for x in list(model.parameters())[0:1]:
            #    loss += (x.data.abs()).sum() * 0.001
            #loss += torch.sum(0.001 * (torch.concat([x. flatten() for x in model.chain.parameters()]).abs() ** 0.5))
            #loss = loss_fun(y_model.flatten(), target) #+  0.0 *  model.cond_filter.loss            
            loss.backward()            
            optimizer.step()
            
            #pdb.set_trace()
            ddd = data.cpu()
            sss=ddd[ddd[:,0]==3,:]  
            sss2=sss[(sss[:,1:-3] == 1).any(dim=1),:] 
            ok += sss2[:,-1].sum()
            total +=   sss2.shape[0]        
        logging.info(f"P(2|0)={ok/total}")
        #pdb.set_trace()

        
        if track is not None and not track(idx, model, device):
            return


def train_ext(model, sampler, device_name='cuda:0', batch_size=20, num_epochs=1000, lr=0.005, track=None, weight_decay=0.0001):    
    logging.info(f"start training")
    device = torch.device(device_name)
    if device is not None:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_model(model=model, loss_fun=nn.BCELoss(reduction='none'), 
        batch_size=batch_size,
        optimizer=optimizer, sampler=sampler, 
        device=device,num_epochs=num_epochs,
        track = track)
    return model


#############

if False:
    sampler = Sampler(SAMPLES)
    model = create_model()

    def mtrack(idx, model, device):
        inp = torch.tensor([
            [2,10,1,10],
            [2,10,1,20],
            [2,10,0,30],
        ])
        values = model(inp.to(device))
        print(inp, values, sep="\n")
        return True

    #example = sampler.sample(1)
    #aaa = model(example)
    train_ext(model, sampler, track=mtrack)
        

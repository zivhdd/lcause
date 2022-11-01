
import torch
from torch import nn
import logging
import pdb

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

def train_model(model, loss_fun, optimizer, sampler, repeats=100, batch_size=100, 
                num_epochs=100, device=None, track=None, model_loss=None):
    for idx in range(num_epochs):

        logging.info(f"{idx}")    
        model.train()
        for rpt in range(repeats):
            data = sampler.sample(batch_size)
            if device is not None:
                data = data.to(device)
            input = data[:,0:-3]
            weight = data[:,-2]
            target = data[:,-1]
            optimizer.zero_grad()
            
            y_model = model(input)            
            loss = (loss_fun(y_model.flatten(), target) * weight).sum() / weight.sum()

            if model_loss:
                loss += model_loss(model)

            loss.backward()            
            optimizer.step()
                    
        if track is not None and not track(idx, model, device):
            return


def train_ext(model, sampler, device_name='cuda:0', batch_size=20, num_epochs=1000, 
    lr=0.005, track=None, weight_decay=0, **kwargs):    
    logging.info(f"start training")
    device = torch.device(device_name)
    if device is not None:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_model(model=model, loss_fun=nn.BCELoss(reduction='none'), 
        batch_size=batch_size,
        optimizer=optimizer, sampler=sampler, 
        device=device,num_epochs=num_epochs,
        track = track,
        **kwargs)

    return model


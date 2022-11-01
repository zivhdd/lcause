
## replace with import
import os, logging
from att import *
import torch
from event import Event, load_event_list
from sampler import Sampler
from model import  train_ext
from att import EventPredictor

import logview
from collections import defaultdict
from samples import SAMPLES3

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

mode = "ball"
logs = load_event_list(f"../data/{mode}.json")
model_path = f"../models/{mode}.json"
result_path = f"../results/{mode}.html"

if mode == "ball":    
    window = 3
    num_any = 2
    weight_any = 8
    nconds=15
elif mode == "trader":
    logs = load_event_list("trader.json")
    window = 180
    num_any = 2
    weight_any = 50
    nconds = 20
else:
    raise Exception("Unexpected mode")

nevents = max([x.logid for x in logs]) + 1

nparams = 0

logging.info(f"nevents={nevents}; nconds={nconds}; nparms={nparams}")

sampler = Sampler(logs, nevents=nevents, nparams=0, nconds=nconds, window=window, num_any=num_any, filter=None)
sampler.config(num_any=num_any, weight_any=weight_any, with_positive=True)

sampler.sample_any(torch.tensor([3]),1.0, 20)


model = LinearPredExtended(nevents=nevents)


if os.path.isfile(model_path):
    obj = torch.load(model_path)
    model.load_state_dict(obj)
    logging.info("loaded model")
    

def model_loss(model):
    betas = model.betas[0:-1,:]
    beta_loss = (betas.abs()).sum() * 0.001            
    return beta_loss


dlogs = [x.to_dict() for x in logs]


def compile(idx, model, device, window=window):
    model.eval()
    torch.save(model.state_dict(), model_path)
    logging.info(f"compiling {idx}")
    lpos = 0
    probs = defaultdict(list)
    causes = defaultdict(list)
    rep = False
    for idx in range(len(logs)):
        while lpos < idx and logs[lpos].time < logs[idx].time - window:
            lpos += 1
        if lpos >= idx:
            continue
        rpos = idx - 1
        
        conds = ([logs[x].logid for x in range(lpos, idx)] + [sampler.tok_no_event] * sampler.nconds)[0:sampler.nconds]
        
        condidx = [logs[x].logid for x in range(lpos, idx)]

        input = [([logs[idx].logid] + conds) for i in range(idx + 1 - lpos)]

        for ci in range(idx - lpos):
            input[ci + 1][ci + 1] = sampler.tok_no_event
        input = torch.tensor(input)
        values = model(input.to(device)) 

        baseval = values[0].tolist()
        probs[idx] = baseval
        for ci in range(idx - lpos):            
            altval = values[ci+1].tolist()
            causes[idx].append([lpos + ci, altval])

    logview.compile(dlogs, probs, causes, output = result_path)
    return True

train_ext(model, sampler, track=compile, batch_size=100, num_epochs=1000, 
    lr=0.005, weight_decay=0, model_loss=model_loss)






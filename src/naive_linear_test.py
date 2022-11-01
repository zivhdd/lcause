
import logging
from collections import defaultdict
from event import load_event_list
import torch
from sampler import Sampler
import logview
logs = load_event_list("log.json")

window = 3
num_any = 10
weight_any = 8
nconds=10
nevents = max([x.logid for x in logs]) + 1
nparams = 0

torch.manual_seed(0)

sampler = Sampler(logs, nevents=nevents, nparams=0, nconds=nconds, window=window, num_any=num_any)
sampler.config(num_any=num_any, weight_any=weight_any, with_positive=True)

samples = sampler.sample(1000)

model = {}
for evid in range(nevents):
    subset = samples[samples[:,0] == evid, :]
    if subset.numel() == 0:
        continue
    Xpart = X = subset[:, 1:-3]
    X = torch.stack([(Xpart==cond).any(dim=1).int() for cond in range(nevents)]).transpose(0,1).float()
    Y = subset[:,-1]
    XTX = X.transpose(0,1) @ X
    XTX = XTX + torch.eye(XTX.shape[0]) * 0.0001
    XTY = X.transpose(0,1) @ Y
    beta = torch.inverse(XTX) @ XTY
    model[evid] = beta

dlogs = [x.to_dict() for x in logs]

def compile(model, window=window):
    logging.info(f"compiling")
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
        
        evnt = logs[idx].logid
        betas = model.get(evnt)
        if betas is None:
            continue
        condset = set([logs[x].logid for x in range(lpos, idx)])        
        conds = torch.tensor([x in condset for x in range(nevents)]).float()
        prob = (betas * conds).sum().tolist()


        probs[idx] = prob

        for ci in range(lpos, idx):
            delta = betas[logs[ci].logid].tolist()
            causes[idx].append([ci, prob - delta])

        #if idx==35:
        #    stophere
        
    logview.compile(dlogs, probs, causes, output = "linear.html")
    return True

compile(model)

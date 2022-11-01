
## replace with import
import logging
from att import *
import torch
from event import Event, load_event_list
from sampler import Sampler
from model import  train_ext
from att import EventPredictor
from sim import LogType
import logview
from collections import defaultdict
from samples import SAMPLES3

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

if False:
    logs = load_event_list("ball.json")
    window = 3
    num_any = 2
    weight_any = 8
    nconds=15
else:
    logs = load_event_list("trader.json")
    window = 180
    num_any = 2
    weight_any = 50
    nconds = 20

nevents = max([x.logid for x in logs]) + 1

nparams = 0


logging.info(f"nevents={nevents}; nconds={nconds}; nparms={nparams}")

sampler = Sampler(logs, nevents=nevents, nparams=0, nconds=nconds, window=window, num_any=num_any, filter=None)
sampler.config(num_any=num_any, weight_any=weight_any, with_positive=True)

sampler.sample_any(torch.tensor([3]),1.0, 20)




#print("####", sampler.sample(1))
#model = EventPredictor(nevents=nevents)
#model = SimplePred2(nevents=nevents, nconds=nconds) ##, emb_dim=40, depth=8)#, depth=6, emb_dim=6)
#model = TransPred(nevents=nevents, nconds=nconds)
#model = create_model(nevents=nevents, nparams=nparams, nconds=nconds)
model = LinearPredExtended(nevents=nevents)

if False:
    obj = torch.load("model2.obj")
    #obj["inner.0.weight"][:] = 0.0
    #obj["inner.0.weight"][:, 6] = 6.0
    #obj["inner.0.bias"][:] = -5.0
    model.load_state_dict(obj)
    logging.info("loaded model")
    


def dep(model, device, event, cond):
    out = model.cond_filter(torch.tensor([[event, cond]], device=device), return_weight=True)
    return (out[0,0].abs().tolist())


def track(idx, model, device):
    print("### C(TASSERT | REJECTED)", dep(model, device, LogType.TICKER_ASSERT, LogType.ORDER_REJECTED))
    print("### C(REJECTED | TASSERT)", dep(model, device, LogType.ORDER_REJECTED, LogType.TICKER_ASSERT))
    print("### C(STOP | TASSERT)", dep(model, device, LogType.STOP_NODES, LogType.TICKER_ASSERT))
    tab = ([[dep(model, device, eventid, condid) for condid in range(nevents)] for eventid in range(nevents)])
    print("\n".join([" ".join(map(lambda x: ("%5s" % f"{(x):.2f}"), ty)) for ty in tab]) )
    #print(tab)
    return True

dlogs = [x.to_dict() for x in logs]

DEBUG = False

def compile(idx, model, device, window=window):
    model.eval()
    torch.save(model.state_dict(), "model2.obj")
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
        #input = [([] + conds) for i in range(idx + 1 - lpos)]
        for ci in range(idx - lpos):
            input[ci + 1][ci + 1] = sampler.tok_no_event
        input = torch.tensor(input)
        values = model(input.to(device)) 
        #print(input, values, sep="\n")

        baseval = values[0].tolist()
        probs[idx] = baseval
        for ci in range(idx - lpos):            
            altval = values[ci+1].tolist()
            causes[idx].append([lpos + ci, altval])

        if DEBUG and idx == 9:
            pdb.set_trace()
    #print("ok")
    logview.compile(dlogs, probs, causes, output = "progress.html")
    return True



def count(aaa):
    ddd=defaultdict(int)
    for idx in aaa:
        ddd[idx] += 1
    return dict(ddd)

train_ext(model, sampler, track=compile, batch_size=100, num_epochs=1000, lr=0.005, weight_decay=0)
#print(count(torch.round(((sampler.rand_time(100000)+300)/1000)).int().tolist()))






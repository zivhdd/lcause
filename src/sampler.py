import torch
from dataclasses import dataclass
import pdb

def shiftl(x, fill):
    rv = x.roll(-1)
    rv[-1] = fill
    return rv

def shiftr(x, fill):
    rv = x.roll(1)
    rv[0] = fill
    return rv 

class Sampler(object):

    def __init__(self, events, window=5, nparams=1, nevents=5,
        nconds=1, with_positive=True, num_any=100, filter=None):

        ## 
        self.nparams = nparams
        self.nconds = nconds
        self.events = events
        self.tok_no_event = nevents + 1

        self.no_event = torch.tensor([self.tok_no_event] + self.extend_params([]))
        self.window = window

        ## default sampling values
        self.with_positive = with_positive
        self.num_any = num_any

        ## times, data
        self.data = torch.stack(
            [torch.tensor([x.logid] + self.extend_params(x.params)) for x in events])
        
        self.times = torch.tensor([x.time for x in events])

        ## helper memeber for randomization
        right_pad = window
        left_pad = 0

        self.pos = torch.arange(len(self.events))                
        if filter is None:
            self.filtered_pos = self.pos
        else:            
            self.filtered_pos = self.pos[self.data[:,0] == filter]

        ends = torch.min(
            shiftl(self.times, self.times[-1] + right_pad),
            self.times + right_pad)

        starts = torch.max(
            shiftr(ends, self.times[0] - left_pad),
            self.times - left_pad)
        #starts = self.times
        
        deltas = ends - starts
        self.start_interval = starts
        self.cumtime = deltas.cumsum(0)
        self.total_time = deltas.sum()
        
        self.cond_fill = torch.tensor([self.tok_no_event] * nconds)
        self.config()


    def config(self, with_positive=True, num_any=10, weight_any=1.0):
        self.with_positive = with_positive
        self.num_any = num_any
        self.weight_any = weight_any

    def rand_time(self, size=2):        
        rnd = torch.rand(size) * self.total_time
        idx = (rnd.unsqueeze(1) >= self.cumtime).sum(dim=1) 
        base = torch.concat((self.cumtime, torch.zeros(1))).roll(1)
        delta = rnd - base[idx]
        times = self.start_interval[idx] + delta
        return times


    def extend_params(self, x):
        if len(x) >= self.nparams:
            return x[0:self.nparams]
        return x + [0] * (self.nparams - len(x))

    def sample(self, num_events):
        samples = []
        for idx in range(num_events):
            samples += self.sample_events(
                with_positive=self.with_positive, 
                num_any=self.num_any, weight_any=self.weight_any)
        
        samples = torch.stack(samples, dim=0)
        perm =  torch.randperm(samples.shape[0])
        return samples[perm, ...]


    def sample_events(self, with_positive=True, num_any=10, weight_any=1.0):
        idx = self.filtered_pos[torch.randint(0, len(self.filtered_pos), (1,))]
        

        event = self.data[idx,:].squeeze(0)            

        samples = self.sample_any(event, weight_any, num_any)        
        if with_positive:            
            selection = torch.logical_and(
                self.times >= self.times[idx] - self.window, 
                self.pos < idx)

            conds = self.create_conds(selection)        
            ## event, conds, weight, target
            positive = torch.concat((event, conds, self.times[idx], torch.ones(2)))
            samples.append(positive)        

        return samples
    
    def create_sample(self, event, barrier, weight):
        selection = torch.logical_and(
            self.times >= barrier - self.window,
            self.times <= barrier + self.window)

        matches = (self.data[:,0] == event[0])
        target_crit = torch.logical_and(matches, selection)

        if torch.any(target_crit):
            candidates = self.pos[target_crit]
            evidx = candidates[torch.randint(0, candidates.numel(), (1,))]
            cond_selection = torch.logical_and(
                selection,
                self.pos < evidx
                )            
            target = 1.0
        else:
            cond_selection = torch.logical_and(
                selection,
                self.times <= barrier)
            target = 0.0

        conds = self.create_conds(cond_selection)
        return (torch.concat((event[0:1], conds, torch.tensor([barrier, weight, target]))))


    def create_conds(self, selection):
        return torch.concat((self.data[selection,0], self.cond_fill))[0:self.nconds]

    def sample_any(self, event, weight, size):
        samp_times = self.rand_time(size).tolist()
        return [self.create_sample(event, barrier, weight) for barrier in samp_times]
        

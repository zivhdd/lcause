
from event import Event
import torch
import random, heapq, logging

class Reactor(object):
    def __init__(self):
        self.timers = []
        self.time = 0
        self.actors = []
        self.logs = []

    def notify(self, event, delay=0):
        self.schedule(delay, lambda: self.notify_i(event))

    def notify_i(self, event):
        for actor in self.actors:
            actor.handle(event)
    
    def log(self, logid, message, params=[]):
        event =  Event(self.time, logid, params=params, text=message)
        self.logs.append(event)
        logging.debug(f"Event: {event}")

    def schedule(self, delta, clb):
        heapq.heappush(self.timers, (self.time + delta, clb))

    def add(self, actor):
        self.actors.append(actor)
        actor.attach(len(self.actors), self)

    def simulate(self, until_time):
        while self.time < until_time and self.timers:
            time, clb = heapq.heappop(self.timers)
            self.time = time
            clb()

    def simulate_for(self, duration):
        self.simulate(self.time + duration)
            
class Actor(object):

    def __init__(self):
        self.poll_mean = 1.0
        self.poll_std = 0.2
        self.poll_min = 0.5

    def attach(self, id, reactor):
        self.id = id
        self.reactor = reactor
        self.attached()
        self.schedule_poll()

    def rndtime(self, mean=1, std=1, min=0):
        return max(torch.normal(self.poll_mean, self.poll_std, (1,)).tolist()[0], self.poll_min)

    def withprob(self, prob):
        return (random.random() < prob)

    def schedule_poll(self):
        interval = self.rndtime(self.poll_mean, self.poll_std, self.poll_min)
        self.reactor.schedule(interval, self.poll)

    def poll(self):
        self.do_poll()
        self.schedule_poll()

    def do_poll(self):
        pass

    def attached(self):
        pass

    def handle(self, event):
        pass

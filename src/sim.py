

from tabnanny import verbose
from event import Event
import torch
import random, heapq
import pickle
import logging
import json

class LogType:
    START_NODES = 0
    NODE_STARTED = 1
    NODE_HEARTBEAT = 2
    TICKER_ASSERT = 3
    STOP_NODES = 4
    NODE_STOPPED = 5
    ORDER_REJECTED = 6
    MARKET_CONDITIONS_A = 7
    MARKET_CONDITIONS_B = 8
    UNEXPECTED_MESSAGE = 9
    STALE_FEED = 10

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
        reactor.schedule(interval, self.poll)

    def poll(self):
        self.do_poll()
        self.schedule_poll()

    def do_poll(self):
        pass

    def attached(self):
        pass

    def handle(self, event):
        pass

class Trader(Actor):

    def __init__(self):
        super().__init__()
        self.session = 0
        self.active = False
        self.last_heartbeat = 0
        self.heartbeat_period = 5 * 60

    def attached(self):
        pass

    def do_poll(self):
        if not self.active:
            return 
        if self.withprob(1/(5*60)):
            self.randlog()

        if self.withprob(1/(30*60)):
            self.ticker_assert()  

        if self.withprob(1/(10*60)):
            self.reactor.log(LogType.ORDER_REJECTED, "Order rejected")
            if self.withprob(0.7):
                self.ticker_assert()


    def ticker_assert(self):
        self.reactor.log(LogType.TICKER_ASSERT, "TickerAssert")
        self.reactor.notify({"type":"ticker-assert"})

    def randlog(self):
        cand = [
            (LogType.MARKET_CONDITIONS_A, "Special Market Conditions A"),
            (LogType.MARKET_CONDITIONS_B, "Special Market Conditions B"),
            (LogType.UNEXPECTED_MESSAGE, "Unexpected Message"),
            (LogType.STALE_FEED, "Stale Feed"),
        ]

        self.reactor.log(*(random.sample(cand, 1)[0]))

    def handle(self, event):        
        etype = event.get("type")

        if etype == "start-nodes" and not self.active:
            self.active = True
            self.reactor.log(LogType.NODE_STARTED, "Node started")

        if etype == "stop-nodes" and  self.active:
            self.active = False
            self.reactor.notify({"type":"node-stopped"})
            self.reactor.log(LogType.NODE_STOPPED, "Node stopped")

class Operator(Actor):

    def attached(self):
        self.reactor.schedule(self.rndtime(3.0,1.0), self.start_all)

    def start_all(self):
        self.reactor.log(LogType.START_NODES, "USER: Starting nodes")
        self.reactor.notify({"type":"start-nodes"})

    def stop_all(self):
        self.reactor.log(LogType.STOP_NODES, "USER: Stopping nodes")
        self.reactor.notify({"type":"stop-nodes"}, self.rndtime(5))

    def handle(self, event):        
        etype = event.get("type")

        if etype == "ticker-assert" and self.withprob(0.5):
            self.reactor.schedule(self.rndtime(60,30,10), self.stop_all)

        if etype == "node-stopped":
            self.reactor.schedule(self.rndtime(60,30,10), self.start_all)
         

class SimpleSim(Actor):

    def __init__(self):
        super().__init__()
        self.poll_mean = 5


    def do_poll(self):        
        if self.withprob(0.5):
            return
        
        self.reactor.log(1, "Whistle blows (round)")
        if self.withprob(0.5):
            self.reactor.log(2, "Birds fly away")
        if self.withprob(0.7):
            self.reactor.log(3, "Kid 1 throws the ball")            
            if self.withprob(0.3):
                self.reactor.log(4, "Kid 2 catches the ball")
                if self.withprob(0.4):
                    self.reactor.log(5, "The kids on the bench are cheering")
            elif self.withprob(0.3):                
                self.reactor.log(6, "The ball hits the trash can")
                if self.withprob(0.6):
                    self.reactor.log(7, "A cat comes out of the trash")
                if self.withprob(0.6):
                    self.reactor.log(8, "The ball rolls away from the trash")
                    if self.withprob(0.7):
                        self.reactor.log(14, "The kid picks up the ball")

            else:
                self.reactor.log(9, "The ball hits a window")
                if self.withprob(0.4):
                    self.reactor.log(10, "The window breaks")
                if self.withprob(0.6):
                    self.reactor.log(12, "The neighbour comes out of the house yelling")
                    if self.withprob(0.6):
                        self.reactor.log(13, "The kids run away")
                    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simulation')
    parser.add_argument('--period', type=int, default = 3600)
    parser.add_argument('--output', type=str, default = None)
    parser.add_argument('--verbose', action='store_true', default = False)
    parser.add_argument('--mode', type=str, default = "trader")
    args = parser.parse_args()    
    print(args)

    logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=(logging.DEBUG if args.verbose else logging.INFO))

    reactor = Reactor()

    if args.mode == "trader":
        reactor.add(Trader())    
        reactor.add(Operator())
    else:
        reactor.add(SimpleSim())

    reactor.simulate_for(args.period)   

    if args.output is not None:
        with open(args.output, "tw") as ofile:
            for log in reactor.logs:
                ofile.write(log.to_json() + "\n")
    



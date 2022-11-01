import pickle, logging, json, random
from event import Event
from simulation import Reactor, Actor

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
         

class BallGameSim(Actor):

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
        reactor.add(BallGameSim())

    reactor.simulate_for(args.period)   

    if args.output is not None:
        with open(args.output, "tw") as ofile:
            for log in reactor.logs:
                ofile.write(log.to_json() + "\n")
    



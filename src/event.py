import json

class Event(object):

    def __init__(self, time=0, logid=0, params=[], text=""):
        self.time = time
        self.logid = logid
        self.params = params
        self.text = text

    def __str__(self):
        return f"{self.time:.1f}: {self.logid} {self.params} :: {self.text}"

    def __repr__(self):
        return f"Event(time={self.time},  logid={self.logid}, {self.params})"

    def to_dict(self):
        return dict(time=self.time, logid=self.logid, params=self.params, text=self.text)

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(line):
        obj = json.loads(line)
        return Event(time=obj["time"], logid=obj["logid"], params=obj["params"], text=obj["text"])

def load_event_list(path):
    with open(path,"tr") as ifile:
        return [Event.from_json(x) for x in ifile]


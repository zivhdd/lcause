
import json
import random
import logview

path = 'log.json'
with open(path,"tr") as ifile:
    logs = [json.loads(x) for x in ifile]
probs = {}
causes = {}

def lpush(obj, idx, value):
    item = obj.get(idx)
    if item is None:
        item = []
        obj[idx] = item
    item.append(value)


for idx in range(len(logs)):
    for icond in range(random.randint(0,5)):
        offset = random.randint(1,5)
        if idx < offset:
            continue
        lpush(probs, idx, [idx-offset, random.random()])
        if icond == 0:
            lpush(causes, idx, [idx-offset, random.random()])
            
logview.compile(logs, probs, causes, output = "stam.html")


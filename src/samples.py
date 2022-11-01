from event import Event


def compile_samples(spec):
    time = 0
    basic_delta = 0.1
    result = []
    idx = 0
    while idx < len(spec):
        item = spec[idx]
        time_delta = item.get("time_delta", basic_delta)
        time += time_delta
        logid = item.get("logid")        
        text = item.get("text", "Unknown")
        text = f"{text} ({logid})"
        params = item.get("params",[])

        if logid is None: 
            idx += 1
            continue
        repeat = item.get("repeat", 1)
        for i in range(repeat):
            time += time_delta
            result.append(Event(logid=logid, time=time, params=params, text=text))
        idx += 1
    return result
            

SAMPLES = compile_samples([

    ##
    {"logid" : 0, "param" : 101, "time_delta": 10},
    {"logid" : 1, "param" : 2, "repeat":1}, 
    {"logid" : 2, "param" : 2, "repeat":1}, 

    {"logid" : 0, "param" : 102, "time_delta": 10},

    ##
    {"logid" : 0, "param" : 103, "time_delta": 10},
    {"logid" : 1, "param" : 3, "repeat":5},
    {"logid" : 2, "param" : 3, "repeat":1}, 

    {"logid" : 0, "param" : 104, "time_delta": 10},

    ##
    {"logid" : 0, "param" : 105, "time_delta": 10},
    {"logid" : 1, "param" : 2, "repeat":3},
    {"logid" : 2, "param" : 2, "repeat":1}, 

    {"logid" : 0, "param" : 106, "time_delta": 10},

    ## (extra)
    {"logid" : 0, "param" : 107, "time_delta": 10},
    {"logid" : 1, "param" : 1, "repeat":1},
    {"logid" : 2, "param" : 1, "repeat":1}, 


    {"logid" : 0, "param" : 108, "time_delta": 10},
    {"logid" : 2, "param" : 5, "repeat":1}, 

    ##
    {"logid" : 0, "param" : 109, "time_delta": 10},
    {"logid" : 1, "param" : 7, "repeat":1},
    {"logid" : 2, "param" : 7, "repeat":1}, 

    {"logid" : 0, "param" : 110, "time_delta": 10},
    
])

SAMPLES1 = compile_samples([
    {"logid" : 1, "param" : 2, "repeat":1, "time_delta":10}, 
    {"logid" : 0, "param" : 101, "time_delta": 0.1},
    {"logid" : 2, "param" : 2, "repeat":1}, 

    
    {"logid" : 1, "param" : 3, "repeat":1, "time_delta": 10}, 
    {"logid" : 0, "param" : 102, "time_delta": 0.10},    
    {"logid" : 2, "param" : 3, "repeat":1}, 


    {"logid" : 1, "param" : 2, "repeat":1, "time_delta": 10}, 
    {"logid" : 0, "param" : 103, "time_delta": 0.10, "repeat":20},
    #{"logid" : 4, "param" : 101, "time_delta": 10},

])

SAMPLES3 = compile_samples([

    {"logid" : 0, "text": "start", "time_delta": 1000}, 

    {"logid" : 0, "text": "start", "time_delta": 1000}, 
    {"logid" : 1, "text": "failue"}, 

    {"logid" : 0, "text": "start", "time_delta": 1000}, 
    {"logid" : 1, "text": "failue"}, 
    {"logid" : 2, "text": "crash"}, 

    {"logid" : 0, "text": "start", "time_delta": 1000}, 
    {"logid" : 1, "text": "failue"}, 
    {"logid" : 2, "text": "crash"}, 


    {"logid" : 0, "text": "start", "time_delta": 1000}, 
    {"logid" : 1, "text": "failue"}, 
    #{"logid" : 2, "text": "crash"}, 
    {"logid" : 3, "text": "reject"},  

    {"logid" : 0, "text": "start", "time_delta": 1000}, 
    {"logid" : 1, "text": "failue"},
    {"logid" : 3, "text": "reject"},  

])

#for x in SAMPLES:
#    print(x)



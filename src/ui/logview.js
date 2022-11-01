

function LogViewer(divID) {
    
    this.logViewDivID =  (divID || "logview")
    this.selection = 10
    this.probs = {}
    this.causes = {}
    deps = {}

    var conv = function(src, dest) {
        //console.log(src)
        for (const [idx, list] of Object.entries(src)) {
            if (deps[idx] == undefined) {
                deps[idx] = []
            }

            for (const item of list) {
                deps[idx].push(item[0])
                dest[idx + ":" + item[0]] = item[1]
            }    
        }
    }
    
    conv(CAUSES, this.causes)
    this.probs = PROBS
    this.deps = deps
}


function lstyle(elm, props, text) {
    elm.classed("logtd", true).
        classed("logtdodd", props.isodd).
        classed("selected", props.selected).
        classed("prob", props.has_prob).
        classed("cause", props.has_cause)

    if (props.cause_factor != undefined) {
        elm.style("background-color", d3.hsl(130, 1 * props.cause_factor, 0.8).hex())
    }   

    if (text) {
        elm.text(text)
    }

    return elm;
}

LogViewer.prototype.refresh = function() {

    dirty = [this.selection, this.prev_selection].concat(this.deps[this.selection] || []).concat(this.deps[this.prev_selection] || [])
    this.prev_selection = this.selection
    for (const lino of dirty) {
        tr = d3.select("#line" + lino)
        tr.html("")
        this.updateLine(tr, lino)
    }
}

LogViewer.prototype.updateLine = function(tr, idx) {
    let self = this
    let current = idx
    let log = LOGS[current]

    isodd = (idx % 2 == 1)
    selected = (this.selection == idx)
    prob = this.probs[idx]
    has_prob = (prob != undefined)

    cause = this.causes[(this.selection + ":" + idx)]
    has_cause = (cause != undefined)

    props = {"selected": selected, "isodd":isodd, "has_prob":false, "has_cause":has_cause, "cause_factor":undefined}
    cause_desc = ""
    prob_desc = ""

    if (has_prob) {
        prob_desc = prob.toFixed(3)
    }

    if (has_cause) {
        cause_desc = cause.toFixed(3)
    }

    tr.on("click", function() {
        self.selection = current
        self.refresh()
    })

    var cause_factor = undefined
    if (has_cause) {            
        event_prob = this.probs[this.selection]
        if (event_prob != undefined) {
            cause_factor = Math.max(Math.min(event_prob - cause, 1),0)
        }
    }

    if (!selected && cause_factor != undefined) {
        props["cause_factor"] = cause_factor
    }

    lstyle(tr.append("td"), props, idx)
    lstyle(tr.append("td"), props, log["time"].toFixed(2))
    lstyle(tr.append("td"), props, "[" + log["logid"] + "] " + log["text"])
    
    if (selected) {
        lstyle(tr.append("td"), {...props, "has_cause":false, "has_prob":has_prob}, prob_desc).style("width", "10%")
    } else {
        elm = lstyle(tr.append("td"), {...props}, cause_desc)
        //elm.style("background-color", d3.hsl(130, 1 * cause_factor, 0.8).hex())
    }
    
}


LogViewer.prototype.draw = function() {
    var self = this
    var base = d3.select("#" + this.logViewDivID)
    base.selectAll("*").remove()

    d3.select("body").on("keydown", function(event) {
        if (event.key == "ArrowDown") {
            self.selection = Math.min(self.selection + 1, LOGS.length -1)
        } if (event.key == "ArrowUp") {
            self.selection = Math.max(self.selection - 1, 0)
        }
        self.refresh()
        iid = "#line" + self.selection
        
        window.setTimeout(function() {
            gap = 200
            pos = d3.select(iid).node().offsetTop
            if (window.scrollY >  pos - gap) {
                window.scrollTo(0, pos - gap)
            } else if (window.scrollY + window.innerHeight < pos + gap) {
                window.scrollTo(0, pos + gap - window.innerHeight)
            }
        }, 0)
        
    })

    var tbl = base.append("table").classed("logtable", true)
    var idx = 0
    for (var idx = 0; idx < LOGS.length; ++idx) {
    
        tr = tbl.append("tr").attr("id", "line"+idx)
        this.updateLine(tr, idx)

    }      
    this.prev_selection = this.selection  
}


var VIEWER = 0

var arrow_keys_handler = function(e) {
    switch(e.code){
        case "ArrowUp": case "ArrowDown": case "ArrowLeft": case "ArrowRight": 
            case "Space": e.preventDefault(); break;
        default: break; // do not block other keys
    }
};
window.addEventListener("keydown", arrow_keys_handler, false);

window.onload = function() {    
    viewer = new LogViewer()
    viewer.draw()
    VIEWER = viewer
}

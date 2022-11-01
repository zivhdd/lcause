import json

class Config:
    template = "ui/template.html"
    style = "ui/styles.css"
    js = "ui/logview.js"

def read_file(path):
    with open(path, "tr") as fh:
        return fh.read()

def json_relations(name, rel):
    return (name + " = " + 
        json.dumps({f"{key[0]}:{key[1]}":value for key, value in rel.items()}) + 
        ";\n")

def compile(logs, probs, causes, cfg=Config, output=None):
    template = read_file(cfg.template)
    styles = read_file(cfg.style)
    jscode = read_file(cfg.js)
    script = (
        f"LOGS = {json.dumps(logs)} ;\n" +
        f"PROBS = {json.dumps(probs)} ;\n" +
        f"CAUSES = {json.dumps(causes)} ;\n" +
        jscode
    )
    content = template.replace("STYLE", styles).replace("SCRIPT", script)

    if output is not None:
        with open(output, "tw") as ofile:
            ofile.write(content)

    return content


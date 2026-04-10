import json
import os

def load_all_metrics():
    with open("metrics/kan_metrics.json") as f:
        kan = json.load(f)

    with open("metrics/googlenet_metrics.json") as f:
        gnet = json.load(f)

    return {
        "kan": kan,
        "googlenet": gnet
    }

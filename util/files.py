import json


def save_dict_to_json(d, filename):
    with open(filename, 'w') as f:
        json.dump(d, f)

def load_dict_from_json(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
        return d

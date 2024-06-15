import json


def read_colors_from_file(path):
    with open(path, 'rt') as fh:
        data = json.load(fh)
    data = {
        k: tuple(v) for k, v in data.items()
    }
    return data


def write_colors_to_file(colors, path):
    with open(path, 'wt') as fh:
        json.dump(colors, fh, sort_keys=True, indent=2)

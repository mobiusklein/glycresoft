import hjson
import os

from .peptide_modification import (load_modification_rule, save_modification_rule)


HOME_DIR = os.path.expanduser("~")
USER_CONFIG_PATH = os.path.join(HOME_DIR, ".glycresoft-cfg.hjson")

HAS_CONFIG = os.path.exists(USER_CONFIG_PATH)

DEFAULT_CONFIG = {
    "peptide_modifications": {},
    "glycan_modifications": {}
}

_CURRENT_CONFIG = None

if not HAS_CONFIG:
    hjson.dump(DEFAULT_CONFIG, open(USER_CONFIG_PATH, 'w'))


def process(config):
    for key, value in config['peptide_modifications'].items():
        load_modification_rule(value)


def get_configuration():
    global _CURRENT_CONFIG
    if _CURRENT_CONFIG is None:
        _CURRENT_CONFIG = hjson.load(open(USER_CONFIG_PATH))
    process(_CURRENT_CONFIG)
    return _CURRENT_CONFIG


def set_configuration(obj):
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = None
    hjson.dump(obj, hjson.load(open(USER_CONFIG_PATH, 'w')))
    return get_configuration()


def add_user_modification_rule(rule):
    serialized = save_modification_rule(rule)
    config = get_configuration()
    config['peptide_modifications'][serialized['full_name']] = serialized
    set_configuration(config)
    return load_modification_rule(serialized)


get_configuration()

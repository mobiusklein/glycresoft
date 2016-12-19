import hjson
import os

from .peptide_modification import (load_modification_rule, save_modification_rule)
from .substituent import (load_substituent_rule, save_substituent_rule)


HOME_DIR = os.path.expanduser("~")
USER_CONFIG_PATH = os.path.join(HOME_DIR, ".glycresoft-cfg.hjson")

HAS_CONFIG = os.path.exists(USER_CONFIG_PATH)

DEFAULT_CONFIG = {
    "version": 0.2,
    "peptide_modifications": {},
    "glycan_modifications": {},
    "substituent_rules": {}
}

_CURRENT_CONFIG = None

if not HAS_CONFIG:
    hjson.dump(DEFAULT_CONFIG, open(USER_CONFIG_PATH, 'w'))


def process(config):
    for key, value in config['peptide_modifications'].items():
        load_modification_rule(value)

    for key, value in config["substituent_rules"].items():
        load_substituent_rule(value)


def get_configuration():
    global _CURRENT_CONFIG
    if _CURRENT_CONFIG is None:
        _CURRENT_CONFIG = hjson.load(open(USER_CONFIG_PATH))
    process(_CURRENT_CONFIG)
    return _CURRENT_CONFIG


def set_configuration(obj):
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = None
    hjson.dump(obj, open(USER_CONFIG_PATH, 'w'))
    return get_configuration()


def add_user_modification_rule(rule):
    serialized = save_modification_rule(rule)
    config = get_configuration()
    config['peptide_modifications'][serialized['full_name']] = serialized
    set_configuration(config)
    return load_modification_rule(serialized)


def add_user_substituent_rule(rule):
    serialized = save_substituent_rule(rule)
    config = get_configuration()
    config['substituent_rules'][serialized['name']] = serialized
    set_configuration(config)
    return load_substituent_rule(serialized)


try:
    get_configuration()
except:
    set_configuration(DEFAULT_CONFIG)

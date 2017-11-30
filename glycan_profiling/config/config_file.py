import hjson
import os
import platform
import click

from .peptide_modification import (load_modification_rule, save_modification_rule)
from .substituent import (load_substituent_rule, save_substituent_rule)

from psims.controlled_vocabulary import controlled_vocabulary as cv


CONFIG_DIR = click.get_app_dir("glycresoft")
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

if platform.system().lower() != 'windows':
    os.environ["NOWAL"] = "1"

_mpl_cache_dir = os.path.join(CONFIG_DIR, 'mpl')

if not os.path.exists(_mpl_cache_dir):
    os.makedirs(_mpl_cache_dir)

os.environ["MPLCONFIGDIR"] = _mpl_cache_dir

CONFIG_FILE_NAME = "glycresoft-cfg.hjson"
USER_CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

cv.configure_obo_store(os.path.join(CONFIG_DIR, "cv"))

HAS_CONFIG = os.path.exists(USER_CONFIG_PATH)

DEFAULT_CONFIG = {
    "version": 0.3,
    "peptide_modifications": {},
    "glycan_modifications": {},
    "substituent_rules": {},
    "environment": {
        "log_file_name": "glycresoft-log",
        "log_file_mode": "a"
    }
}

_CURRENT_CONFIG = None

if not HAS_CONFIG:
    hjson.dump(DEFAULT_CONFIG, open(USER_CONFIG_PATH, 'w'))


def process(config):
    for key, value in config['peptide_modifications'].items():
        load_modification_rule(value)

    for key, value in config["substituent_rules"].items():
        load_substituent_rule(value)


def recursive_merge(a, b):
    for k, v in b.items():
        if isinstance(b[k], dict) and isinstance(a.get(k), dict):
            recursive_merge(a[k], v)
        else:
            a[k] = v


def get_configuration():
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = load_configuration_from_path(USER_CONFIG_PATH)
    if os.path.exists(os.path.join(os.getcwd(), CONFIG_FILE_NAME)):
        local_config = load_substituent_rule(os.path.join(os.getcwd(), CONFIG_FILE_NAME))
        recursive_merge(_CURRENT_CONFIG, local_config)
    return _CURRENT_CONFIG


def load_configuration_from_path(path):
    cfg = hjson.load(open(path))
    process(cfg)
    return cfg


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
except Exception:
    set_configuration(DEFAULT_CONFIG)

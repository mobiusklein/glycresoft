import hjson
import os
import platform
import shutil
import click


from .peptide_modification import (load_modification_rule, save_modification_rule)
from .substituent import (load_substituent_rule, save_substituent_rule)

from psims.controlled_vocabulary import controlled_vocabulary as cv


CONFIG_DIR = click.get_app_dir("glycresoft")

if platform.system().lower() != 'windows':
    os.environ["NOWAL"] = "1"

_mpl_cache_dir = os.path.join(CONFIG_DIR, 'mpl')
cv_store = os.path.join(CONFIG_DIR, 'cv')


def populate_config_dir():
    os.makedirs(CONFIG_DIR)

    # Pre-populate OBO Cache to avoid needing to look these up
    # by URL later
    os.makedirs(cv_store)
    with open(os.path.join(cv_store, 'psi-ms.obo'), 'wb') as fh:
        fh.write(cv._use_vendored_psims_obo().read())
    with open(os.path.join(cv_store, 'unit.obo'), 'wb') as fh:
        fh.write(cv._use_vendored_unit_obo().read())

    os.makedirs(_mpl_cache_dir)


def delete_config_dir():
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)


if not os.path.exists(CONFIG_DIR):
    populate_config_dir()

if not os.path.exists(_mpl_cache_dir):
    os.makedirs(_mpl_cache_dir)

os.environ["MPLCONFIGDIR"] = _mpl_cache_dir

CONFIG_FILE_NAME = "glycresoft-cfg.hjson"
USER_CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

cv.configure_obo_store(os.path.join(CONFIG_DIR, "cv"))

HAS_CONFIG = os.path.exists(USER_CONFIG_PATH)

DEFAULT_CONFIG = {
    "version": 0.4,
    "peptide_modifications": {},
    "glycan_modifications": {},
    "substituent_rules": {},
    "environment": {
        "log_file_name": "glycresoft-log",
        "log_file_mode": "a"
    },
    "xml_huge_tree": True
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
    _CURRENT_CONFIG = load_configuration_from_path(USER_CONFIG_PATH, apply=False)
    local_config_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    if os.path.exists(local_config_path):
        local_config = load_configuration_from_path(local_config_path, apply=False)
        recursive_merge(_CURRENT_CONFIG, local_config)
    process(_CURRENT_CONFIG)
    return _CURRENT_CONFIG


def load_configuration_from_path(path, apply=True):
    cfg = hjson.load(open(path))
    if apply:
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


DEBUG_MODE = bool(os.environ.get("GLYCRESOFTDEBUG"))

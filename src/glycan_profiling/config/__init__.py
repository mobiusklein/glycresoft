
def get_configuration():
    from .config_file import get_configuration
    return get_configuration()


from .config_file import DEBUG_MODE


__all__ = [
    "get_configuration", DEBUG_MODE
]

import logging
from glycan_profiling import task

try:
    logging.basicConfig(level=logging.DEBUG, filename='glycresoft-log', filemode='w',
                        format="%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S")
    logging.captureWarnings(True)
    logger = logging.getLogger("glycresoft")

    task.TaskBase.log_with_logger(logger)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logging.getLogger().addHandler(handler)

    warner = logging.getLogger('py.warnings')
    warner.setLevel("CRITICAL")

except Exception, e:
    logging.exception("Error, %r", e, exc_info=e)
    raise e

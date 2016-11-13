import logging
from glycan_profiling import task
from multiprocessing import current_process

try:
    logging.basicConfig(level=logging.DEBUG, filename='glycresoft-log', filemode='w',
                        format="%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S")
    logger_to_silence = logging.getLogger("deconvolution_scan_processor")
    logger_to_silence.propagate = False
    logging.captureWarnings(True)
    logger = logging.getLogger("glycresoft")

    task.TaskBase.log_with_logger(logger)

    if current_process().name == "MainProcess":
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

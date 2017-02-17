import logging
import codecs
import uuid
from stat import S_IRUSR, S_IWUSR, S_IXUSR, S_IRGRP, S_IWGRP, S_IROTH, S_IWOTH
import os
from logging import FileHandler

from glycan_profiling import task
import multiprocessing
from multiprocessing import current_process


log_multiprocessing = False


def configure_logging(level=logging.DEBUG):
    file_fmter = logging.Formatter(
        "%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        "%H:%M:%S")
    handler = FlexibleFileHandler("glycresoft-log", mode='w')
    handler.setFormatter(file_fmter)
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)

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
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)

        if log_multiprocessing:
            multilogger = multiprocessing.get_logger()
            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            handler.setLevel(level)
            multilogger.addHandler(handler)

    warner = logging.getLogger('py.warnings')
    warner.setLevel("CRITICAL")


permission_mask = S_IRUSR & S_IWUSR & S_IXUSR & S_IRGRP & S_IWGRP & S_IROTH & S_IWOTH


class FlexibleFileHandler(FileHandler):
    def _get_available_file(self):
        basename = current_name = self.baseFilename
        suffix = 0
        while suffix < 2 ** 16:
            if not os.path.exists(current_name):
                break
            elif os.access(current_name, os.W_OK):
                break
            else:
                suffix += 1
                current_name = "%s.%s" % (basename, suffix)
        if suffix < 2 ** 16:
            return current_name
        else:
            return "%s.%s" % (basename, uuid.uuid4().hex)

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        file_name = self._get_available_file()
        if self.encoding is None:
            stream = open(file_name, self.mode)
        else:
            stream = codecs.open(file_name, self.mode, self.encoding)
        return stream

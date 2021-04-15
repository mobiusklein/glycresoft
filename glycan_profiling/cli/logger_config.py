import io
import os
import sys
import uuid
import codecs
import logging
from logging import FileHandler
import warnings
import traceback
import multiprocessing
from multiprocessing import current_process

from six import PY2

from stat import S_IRUSR, S_IWUSR, S_IXUSR, S_IRGRP, S_IWGRP, S_IROTH, S_IWOTH

from glycan_profiling import task
from glycan_profiling.config import config_file

logging_levels = {
    'CRITICAL': 50,
    'DEBUG': 10,
    'ERROR': 40,
    'INFO': 20,
    'NOTSET': 0,
    'WARN': 30,
    'WARNING': 30
}

LOG_FILE_NAME = "glycresoft-log"
LOG_FILE_MODE = 'w'
LOG_LEVEL = 'INFO'

user_config_data = config_file.get_configuration()
try:
    LOG_FILE_NAME = user_config_data['environment']['log_file_name']
except KeyError:
    pass
try:
    LOG_FILE_MODE = user_config_data['environment']['log_file_mode']
except KeyError:
    pass
try:
    LOG_LEVEL = user_config_data['environment']['log_level']
    LOG_LEVEL = str(LOG_LEVEL).upper()
except KeyError:
    pass


LOG_FILE_NAME = os.environ.get("GLYCRESOFT_LOG_FILE_NAME", LOG_FILE_NAME)
LOG_FILE_MODE = os.environ.get("GLYCRESOFT_LOG_FILE_MODE", LOG_FILE_MODE)
LOG_LEVEL = str(os.environ.get("GLYCRESOFT_LOG_LEVEL", LOG_LEVEL)).upper()


log_multiprocessing = False
try:
    log_multiprocessing = bool(user_config_data['environment']['log_multiprocessing'])
except KeyError:
    pass


log_multiprocessing = bool(int(os.environ.get(
    "GLYCRESOFT_LOG_MULTIPROCESSING", log_multiprocessing)))


LOGGING_CONFIGURED = False


class ProcessAwareFormatter(logging.Formatter):
    def format(self, record):
        d = record.__dict__
        try:
            if d['processName'] == "MainProcess":
                d['maybeproc'] = ''
            else:
                d['maybeproc'] = ":%s:" % d['processName']
        except KeyError:
            d['maybeproc'] = ''
        return super(ProcessAwareFormatter, self).format(record)


def configure_logging(level=None, log_file_name=None, log_file_mode=None):
    global LOGGING_CONFIGURED
    # If we've already called this, don't repeat it
    if LOGGING_CONFIGURED:
        return
    else:
        LOGGING_CONFIGURED = True
    if level is None:
        level = logging_levels.get(LOG_LEVEL, "INFO")
    if log_file_name is None:
        log_file_name = LOG_FILE_NAME
    if log_file_mode is None:
        log_file_mode = LOG_FILE_MODE
    file_fmter = ProcessAwareFormatter(
        "%(asctime)s %(name)s:%(filename)s:%(lineno)-4d - %(levelname)s%(maybeproc)s - %(message)s",
        "%H:%M:%S")
    if log_file_mode not in ("w", "a"):
        warnings.warn("File Logger configured with mode %r not applicable, using \"w\" instead" % (
            log_file_mode,))
        log_file_mode = "w"
    handler = FlexibleFileHandler(log_file_name, mode=log_file_mode)
    handler.setFormatter(file_fmter)
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)

    logger_to_silence = logging.getLogger("deconvolution_scan_processor")
    logger_to_silence.propagate = False
    logging.captureWarnings(True)

    logger = logging.getLogger("glycresoft")
    task.TaskBase.log_with_logger(logger)

    status_logger = logging.getLogger("glycresoft.status")
    status_logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s: %(message)s", "%H:%M:%S"))
    status_logger.addHandler(handler)

    if current_process().name == "MainProcess":
        fmt = ProcessAwareFormatter(
            "%(asctime)s %(name)s:%(filename)s:%(lineno)-4d - %(levelname)s%(maybeproc)s - %(message)s", "%H:%M:%S")
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
        stream = LazyFile(self.baseFilename, self.mode, self.encoding)
        return stream


class LazyFile(object):
    def __init__(self, name, mode='r', encoding=None):
        self.name = name
        self.mode = mode
        self.encoding = encoding
        self._file = None

    def _open(self):
        file_name = self._get_available_file()
        if self.encoding is None:
            stream = open(file_name, self.mode)
        else:
            stream = codecs.open(file_name, self.mode, self.encoding)
        self.name = file_name
        self._file = stream
        return self._file

    def read(self, n=None):
        if self._file is None:
            self._open()
        return self._file.read(n)

    def write(self, t):
        if self._file is None:
            self._open()
        return self._file.write(t)

    def close(self):
        if self._file is not None:
            self._file.close()
        return None

    def flush(self):
        if self._file is None:
            return
        return self._file.flush()

    def _get_available_file(self):
        basename = current_name = self.name
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


if hasattr(sys, '_getframe'):
    def currentframe():
        return sys._getframe(3)
else:
    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception()
        except Exception:
            return sys.exc_info()[2].tb_frame.f_back


_srcfile = os.path.normcase(currentframe.__code__.co_filename)


def find_caller(self, stack_info=False, stacklevel=1):
    f = currentframe()
    # On some versions of IronPython, currentframe() returns None if
    # IronPython isn't run with -X:Frames.
    orig_f = f
    stacklevel += 2
    while f and stacklevel > 1:
        f = f.f_back
        stacklevel -= 1
    if not f:
        f = orig_f
    rv = "(unknown file)", 0, "(unknown function)"
    sinfo = None
    i = 0
    while hasattr(f, "f_code"):
        i += 1
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if filename == _srcfile:
            f = f.f_back
            continue
        sinfo = None
        if stack_info:
            sio = io.StringIO()
            sio.write('Stack (most recent call last):\n')
            traceback.print_stack(f, file=sio)
            sinfo = sio.getvalue()
            if sinfo[-1] == '\n':
                sinfo = sinfo[:-1]
            sio.close()
        rv = (os.path.splitext(os.path.basename(co.co_filename))[0].ljust(13)[:13],
              f.f_lineno, co.co_name.ljust(5)[:5])
        break
    if not PY2:
        rv += (sinfo, )
    return rv


logging.Logger.findCaller = find_caller

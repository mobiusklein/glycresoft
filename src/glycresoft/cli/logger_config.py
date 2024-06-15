import os
import re
import sys
import uuid
import codecs
import logging
import warnings
import multiprocessing

from typing import Dict
from logging import FileHandler, LogRecord
from multiprocessing import current_process

from stat import S_IRUSR, S_IWUSR, S_IXUSR, S_IRGRP, S_IWGRP, S_IROTH, S_IWOTH

from ms_deisotope.task.log_utils import LogUtilsMixin as _LogUtilsMixin

from glycresoft import task
from glycresoft.config import config_file

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
    def format(self, record: logging.LogRecord) -> str:
        d: Dict[str, str] = record.__dict__
        try:
            d['filename'] = d['filename'].replace(".py", '').ljust(13)[:13]
        except KeyError:
            d['filename'] = ''
        try:
            if d['processName'] == "MainProcess":
                d['maybeproc'] = ''
            else:
                d['maybeproc'] = "| %s " % d['processName'].replace(
                    "Process", '')
        except KeyError:
            d['maybeproc'] = ''
        return super(ProcessAwareFormatter, self).format(record)


class LevelAwareColoredLogFormatter(ProcessAwareFormatter):
    try:
        from colorama import Fore, Style, init
        init()
        GREY = Fore.WHITE
        BLUE = Fore.BLUE
        GREEN = Fore.GREEN
        YELLOW = Fore.YELLOW
        RED = Fore.RED
        BRIGHT = Style.BRIGHT
        DIM = Style.DIM
        BOLD_RED = Fore.RED + Style.BRIGHT
        RESET = Style.RESET_ALL
    except ImportError:
        GREY = ''
        BLUE = ''
        GREEN = ''
        YELLOW = ''
        RED = ''
        BRIGHT = ''
        DIM = ''
        BOLD_RED = ''
        RESET = ''

    def _colorize_field(self, fmt: str, field: str, color: str) -> str:
        return re.sub("(" + field + ")", color + r"\1" + self.RESET, fmt)

    def _patch_fmt(self, fmt: str, level_color: str) -> str:
        fmt = self._colorize_field(fmt, r"%\(asctime\)s", self.GREEN)
        fmt = self._colorize_field(fmt, r"%\(name\).*?s", self.BLUE)
        # fmt = self._colorize_field(fmt, r"%\(message\).*?s", self.GREY)
        if level_color:
            fmt = self._colorize_field(fmt, r"%\(levelname\).*?s", level_color)
        return fmt

    def __init__(self, fmt, level_color=None, **kwargs):
        fmt = self._patch_fmt(fmt, level_color=level_color)
        super().__init__(fmt, **kwargs)


class ColoringFormatter(logging.Formatter):
    level_to_color = {
        logging.INFO: LevelAwareColoredLogFormatter.GREEN,
        logging.DEBUG: LevelAwareColoredLogFormatter.GREY + LevelAwareColoredLogFormatter.DIM,
        logging.WARN: LevelAwareColoredLogFormatter.YELLOW + LevelAwareColoredLogFormatter.BRIGHT,
        logging.ERROR: LevelAwareColoredLogFormatter.BOLD_RED,
        logging.CRITICAL: LevelAwareColoredLogFormatter.BOLD_RED,
        logging.FATAL: LevelAwareColoredLogFormatter.RED + LevelAwareColoredLogFormatter.DIM,
    }

    _formatters: Dict[int, LevelAwareColoredLogFormatter]

    def __init__(self, fmt: str, **kwargs):
        self._formatters = {}
        for level, style in self.level_to_color.items():
            self._formatters[level] = LevelAwareColoredLogFormatter(
                fmt, level_color=style, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        fmtr = self._formatters[record.levelno]
        return fmtr.format(record)


DATE_FMT = "%H:%M:%S"


class _LogFilter(logging.Filter):
    """Filter out log messages from redundant logger helpers"""

    def filter(self, record: LogRecord) -> bool:
        if record.module == 'log_utils':
            return False
        return super().filter(record)


def get_status_formatter():
    status_terminal_format_string = '[%(asctime)s] %(levelname).1s | %(name)s | %(message)s'
    # status_terminal_format_string = "%(asctime)s %(name)s:%(levelname).1s - %(message)s"

    if sys.stderr.isatty():
        terminal_formatter = ColoringFormatter(
            status_terminal_format_string, datefmt=DATE_FMT)
    else:
        terminal_formatter = logging.Formatter(
            status_terminal_format_string, datefmt=DATE_FMT)
    return terminal_formatter


def make_status_logger(name="glycresoft.status"):
    status_logger = logging.getLogger(name)
    status_logger.propagate = False

    terminal_formatter = get_status_formatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(terminal_formatter)
    status_logger.addHandler(handler)
    return status_logger


def get_main_process_formatter():
    terminal_format_string = (
        "[%(asctime)s] %(levelname).1s | %(name)s | %(filename)s:%(lineno)-4d%(maybeproc)s| %(message)s"
    )
    if sys.stderr.isatty():
        fmt = ColoringFormatter(
            terminal_format_string, datefmt=DATE_FMT)
    else:
        fmt = ProcessAwareFormatter(
            terminal_format_string, datefmt=DATE_FMT)
    return fmt


def make_main_process_logger(name=None, level="INFO"):
    fmt = get_main_process_formatter()
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)
    logging.getLogger(name=name).addHandler(handler)


def make_log_file_logger(log_file_name, log_file_mode, name=None, level="INFO"):
    file_fmter = ProcessAwareFormatter(
        "[%(asctime)s] %(levelname).1s | %(name)s | %(filename)s:%(lineno)-4d%(maybeproc)s | %(message)s",
        DATE_FMT)
    flex_handler = FlexibleFileHandler(log_file_name, mode=log_file_mode)
    flex_handler.setFormatter(file_fmter)
    flex_handler.setLevel(level)
    logger = logging.getLogger(name=name)
    for handler in list(logger.handlers):
        if isinstance(handler, FlexibleFileHandler):
            logger.removeHandler(handler)
    logger.addHandler(flex_handler)


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

    if log_file_mode not in ("w", "a"):
        warnings.warn("File Logger configured with mode %r not applicable, using \"w\" instead" % (
            log_file_mode,))
        log_file_mode = "w"
    logging.getLogger().setLevel(level)

    log_filter = _LogFilter()

    logger_to_silence = logging.getLogger("ms_deisotope")
    logger_to_silence.setLevel(logging.INFO)
    logger_to_silence.addFilter(log_filter)
    logging.captureWarnings(True)

    logger = logging.getLogger("glycresoft")
    # We probably only need to register the base class but let's play it safe
    _LogUtilsMixin.log_with_logger(logger)
    task.LoggingMixin.log_with_logger(logger)
    task.TaskBase.log_with_logger(logger)

    make_status_logger()

    if current_process().name == "MainProcess":
        make_main_process_logger(level=level)

        if log_multiprocessing:
            fmt = get_main_process_formatter()
            multilogger = multiprocessing.get_logger()
            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            handler.setLevel(level)
            multilogger.addHandler(handler)

        if log_file_name:
            make_log_file_logger(log_file_name, log_file_mode, name='glycresoft', level=level)

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
            try:
                stream = codecs.open(file_name, self.mode, self.encoding)
            except LookupError:
                print(f"!! Failed to look up encoding {self.encoding}, falling back to UTF-8")
                stream = open(file_name, self.mode, encoding='utf8')
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

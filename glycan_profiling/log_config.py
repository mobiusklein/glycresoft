from __future__ import print_function
import sys
import codecs
import copy
import logging
import traceback
import warnings
import multiprocessing
from multiprocessing import current_process
import threading
import six
import os
import io
import uuid
from datetime import datetime

from six import PY2

from stat import S_IRUSR, S_IWUSR, S_IXUSR, S_IRGRP, S_IWGRP, S_IROTH, S_IWOTH

try:
    from Queue import Empty
except ImportError:
    from queue import Empty

from glycan_profiling.version import version

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
    log_multiprocessing = bool(
        user_config_data['environment']['log_multiprocessing'])
except KeyError:
    pass


log_multiprocessing = bool(int(os.environ.get(
    "GLYCRESOFT_LOG_MULTIPROCESSING", log_multiprocessing)))


def display_version(print_fn):
    msg = "glycresoft: version %s" % version
    print_fn(msg)


def ensure_text(obj):
    if six.PY2:
        return unicode(obj)
    else:
        return str(obj)


def fmt_msg(*message):
    return u"%s %s" % (ensure_text(datetime.now().isoformat(' ')), u', '.join(map(ensure_text, message)))


def printer(obj, *message):
    print(fmt_msg(*message))


def debug_printer(obj, *message):
    if obj.in_debug_mode():
        print(u"DEBUG:" + fmt_msg(*message))


class ProcessAwareFormatter(logging.Formatter):
    def format(self, record):
        if record.__dict__['processName'] == "MainProcess":
            record.__dict__['maybeproc'] = ''
        else:
            record.__dict__['maybeproc'] = ":%s:" % record.__dict__[
                'processName']
        return super(ProcessAwareFormatter, self).format(record)


# Copied from Py3 logging.handler as suggested by the documentation.
class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.
    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record):
        """
        Enqueue a record.
        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.
        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.
        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also returns the formatted
        # message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info and exc_text attributes, as they are no longer
        # needed and, if not None, will typically not be pickleable.
        msg = self.format(record)
        # bpo-35726: make copy of record to avoid affecting other handlers in the chain.
        record = copy.copy(record)
        record.message = msg
        record.msg = msg
        record.args = None
        record.exc_info = None
        record.exc_text = None
        return record

    def emit(self, record):
        """
        Emit a record.
        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception as err:
            self.handleError(record)


class QueueListener(object):
    """
    This class implements an internal threaded listener which watches for
    LogRecords being added to a queue, removes them and passes them to a
    list of handlers for processing.
    """
    _sentinel = None

    def __init__(self, queue, *handlers):
        """
        Initialise an instance with the specified queue and
        handlers.
        """
        self.queue = queue
        self.handlers = handlers
        self._thread = None
        self.respect_handler_level = True

    def dequeue(self, block):
        """
        Dequeue a record and return it, optionally blocking.
        The base implementation uses get. You may want to override this method
        if you want to use timeouts or work with custom queue implementations.
        """
        return self.queue.get(block)

    def start(self):
        """
        Start the listener.
        This starts up a background thread to monitor the queue for
        LogRecords to process.
        """
        self._thread = t = threading.Thread(target=self._monitor)
        t.daemon = True
        t.start()

    def prepare(self, record):
        """
        Prepare a record for handling.
        This method just returns the passed-in record. You may want to
        override this method if you need to do any custom marshalling or
        manipulation of the record before passing it to the handlers.
        """
        return record

    def handle(self, record):
        """
        Handle a record.
        This just loops through the handlers offering them the record
        to handle.
        """
        record = self.prepare(record)
        for handler in self.handlers:
            if not self.respect_handler_level:
                process = True
            else:
                process = record.levelno >= handler.level
            if process:
                handler.handle(record)

    def _monitor(self):
        """
        Monitor the queue for records, and ask the handler
        to deal with them.
        This method runs on a separate, internal thread.
        The thread will terminate if it sees a sentinel object in the queue.
        """
        q = self.queue
        has_task_done = hasattr(q, 'task_done')
        while True:
            try:
                record = self.dequeue(True)
                if record is self._sentinel:
                    if has_task_done:
                        q.task_done()
                    break
                self.handle(record)
                if has_task_done:
                    q.task_done()
            except Empty:
                break

    def enqueue_sentinel(self):
        """
        This is used to enqueue the sentinel record.
        The base implementation uses put_nowait. You may want to override this
        method if you want to use timeouts or work with custom queue
        implementations.
        """
        self.queue.put_nowait(self._sentinel)

    def stop(self):
        """
        Stop the listener.
        This asks the thread to terminate, and then waits for it to do so.
        Note that if you don't call this before your application exits, there
        may be some records still left on the queue, which won't be processed.
        """
        self.enqueue_sentinel()
        self._thread.join()
        self._thread = None


class CallInterval(object):
    """Call a function every `interval` seconds from
    a separate thread.

    Attributes
    ----------
    stopped: threading.Event
        A semaphore lock that controls when to run `call_target`
    call_target: callable
        The thing to call every `interval` seconds
    args: iterable
        Arguments for `call_target`
    interval: number
        Time between calls to `call_target`
    """

    def __init__(self, interval, call_target, *args):
        self.stopped = threading.Event()
        self.interval = interval
        self.call_target = call_target
        self.args = args
        self.thread = threading.Thread(target=self.mainloop)
        self.thread.daemon = True

    def mainloop(self):
        logger = logging.getLogger("glycresoft.task")
        while not self.stopped.wait(self.interval):
            try:
                self.call_target(*self.args)
            except Exception as e:
                logger.exception("An error occurred in %r", self, exc_info=e)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()


class MessageSpooler(object):
    """An IPC-based logging helper

    Attributes
    ----------
    halting : bool
        Whether the object is attempting to
        stop, so that the internal thread can
        tell when it should stop and tell other
        objects using it it is trying to stop
    handler : Callable
        A Callable object which can be used to do
        the actual logging
    message_queue : multiprocessing.Queue
        The Inter-Process Communication queue
    thread : threading.Thread
        The internal listener thread that will consume
        message_queue work items
    """

    def __init__(self, handler):
        self.handler = handler
        self.message_queue = multiprocessing.Queue()
        self.halting = False
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def handle_message(self, message):
        if isinstance(self.handler, logging.Logger):
            process = message.levelno >= self.handler.level
            if process:
                self.handler.handle(message)
        else:
            self.handler(*message)

    def run(self):
        while not self.halting:
            try:
                message = self.message_queue.get(True, 2)
                self.handle_message(message)
            except Empty:
                continue
            except Exception as err:
                print("Error:", type(err), err)
                continue

    def stop(self):
        self.halting = True
        self.thread.join()

    def sender(self):
        return MessageSender(self.message_queue)


class MessageSender(object):
    """A simple callable for pushing objects into an IPC
    queue.

    Attributes
    ----------
    queue : multiprocessing.Queue
        The Inter-Process Communication queue
    """

    def __init__(self, queue):
        self.queue = queue
        self.configured = False
        self.logger = None

    def install(self):
        print("Installing Logging Config")
        configure_logging_multiprocessing(self.queue)
        self.logger = logging.getLogger("glycresoft")
        self.configured = True
        # Force all logging in the current process to flow through this object. This may cause
        # trouble if the queue shuts down.
        # process = current_process()
        # if process.name == "MainProcess":
        #     raise ValueError("Should not be invoked from the main process!")
        LoggingMixin.log_with_logger(self)

    def __call__(self, *message, **kwargs):
        kwargs.setdefault("level", logging.INFO)
        level = kwargs.pop('level')
        if not self.configured:
            self.install()
        self.logger.log(level, *message, **kwargs)

    def log(self, *args, **kwargs):
        return self(*args, **kwargs)

    def info(self, *args, **kwargs):
        kwargs['level'] = logging.INFO
        return self(*args, **kwargs)

    def debug(self, *args, **kwargs):
        kwargs['level'] = logging.DEBUG
        return self(*args, **kwargs)

    def error(self, *args, **kwargs):
        kwargs['level'] = logging.ERROR
        return self(*args, **kwargs)

    def warn(self, *args, **kwargs):
        kwargs['level'] = logging.WARNING
        return self(*args, **kwargs)

    def critical(self, *args, **kwargs):
        kwargs['level'] = logging.CRITICAL
        return self(*args, **kwargs)



permission_mask = S_IRUSR & S_IWUSR & S_IXUSR & S_IRGRP & S_IWGRP & S_IROTH & S_IWOTH


class FlexibleFileHandler(logging.FileHandler):
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


LOGGING_CONFIGURED = False


def make_file_logger_formatter():
    file_fmter = ProcessAwareFormatter(
        "%(asctime)s %(name)s:%(filename)s:%(lineno)-4d - %(levelname)s%(maybeproc)s - %(message)s",
        "%H:%M:%S")
    return file_fmter


def configure_logging_multiprocessing(queue):
    if queue is None:
        raise ValueError("Logging queue cannot be None!")
    handler = QueueHandler(queue)
    handler.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(logging.StreamHandler())
    warner = logging.getLogger('py.warnings')
    warner.setLevel("CRITICAL")


def configure_logging(level=None, log_file_name=None, log_file_mode=None):
    global LOGGING_CONFIGURED
    # If we've already called this, don't repeat it
    if LOGGING_CONFIGURED:
        return
    else:
        LOGGING_CONFIGURED = True
    process = current_process()
    if process.name != "MainProcess":
        return

    if level is None:
        level = logging_levels.get(LOG_LEVEL, "INFO")
    if log_file_name is None:
        log_file_name = LOG_FILE_NAME
    if log_file_mode is None:
        log_file_mode = LOG_FILE_MODE
    file_fmter = make_file_logger_formatter()
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
    LoggingMixin.log_with_logger(logger)

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


class LoggingMixin(object):
    logger_state = None
    print_fn = printer
    debug_print_fn = debug_printer
    error_print_fn = printer

    @classmethod
    def log_with_logger(cls, logger):
        LoggingMixin.logger_state = logger
        LoggingMixin.print_fn = logger.info
        LoggingMixin.debug_print_fn = logger.debug
        LoggingMixin.error_print_fn = logger.error

    @classmethod
    def log_to_stdout(cls):
        cls.logger_state = None
        cls.print_fn = printer
        cls.debug_print_fn = debug_printer
        cls.error_print_fn = printer

    def log(self, *message):
        self.print_fn(u', '.join(map(ensure_text, message)))

    def debug(self, *message):
        self.debug_print_fn(u', '.join(map(ensure_text, message)))

    def error(self, *message, **kwargs):
        exception = kwargs.get("exception")
        self.error_print_fn(u', '.join(map(ensure_text, message)))
        if exception is not None:
            self.error_print_fn(traceback.format_exc(exception))

    def ipc_logger(self, handler=None):
        if handler is None:
            if self.logger_state is None:
                def _default_closure_handler(message):
                    self.log(message)
                handler = _default_closure_handler
            else:
                handler = self.logger_state
        return MessageSpooler(handler)

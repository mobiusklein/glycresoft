from __future__ import print_function
import os
import logging
import pprint
import time
import traceback
import multiprocessing
import threading

from typing import Any, Generic, List, Optional, TypeVar, Union
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.managers import SyncManager
from datetime import datetime

from queue import Empty
import warnings

import six

from glycresoft.version import version



logger = logging.getLogger("glycresoft.task")

T = TypeVar("T")


def display_version(print_fn):
    msg = "glycresoft: version %s" % version
    print_fn(msg)


def ensure_text(obj):
    if six.PY2:
        return six.text_type(obj)
    else:
        return str(obj)


def fmt_msg(*message):
    return u"%s %s" % (ensure_text(datetime.now().isoformat(' ')), u', '.join(map(ensure_text, message)))


def printer(obj, *message, stacklevel=None):
    print(fmt_msg(*message))


def debug_printer(obj, *message, stacklevel=None):
    if obj.in_debug_mode():
        print(u"DEBUG:" + fmt_msg(*message))


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
        while not self.stopped.wait(self.interval):
            try:
                self.call_target(*self.args)
            except Exception as e:
                logger.exception("An error occurred in %r", self, exc_info=e)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()


class IPCLoggingManager:
    queue: multiprocessing.Queue
    listener: QueueListener

    def __init__(self, queue=None, *handlers):
        if queue is None:
            queue = multiprocessing.Queue()
        if not handlers:
            logger = logging.getLogger()
            handlers = logger.handlers

        self.queue = queue
        self.listener = QueueListener(
            queue, *handlers, respect_handler_level=True)
        self.listener.start()

    def sender(self, logger_name="glycresoft"):
        return LoggingHandlerToken(self.queue, logger_name)

    def start(self):
        self.listener.start()

    def stop(self):
        try:
            self.listener.stop()
        except AttributeError:
            pass


class LoggingHandlerToken:
    queue: multiprocessing.Queue
    name: str
    configured: bool

    def __init__(self, queue: multiprocessing.Queue, name: str):
        self.queue = queue
        self.name = name
        self.configured = False

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        return logger

    def clear_handlers(self, logger: logging.Logger):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        if logger.parent is not None and logger.parent is not logger:
            self.clear_handlers(logger.parent)

    def log(self, *args, **kwargs):
        kwargs.setdefault('stacklevel', 2)
        self.get_logger().info(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        self.log(*args, **kwargs)

    def add_handler(self):
        if self.configured:
            return
        logger = self.get_logger()
        self.clear_handlers(logger)
        handler = QueueHandler(self.queue)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        LoggingMixin.log_with_logger(logger)
        TaskBase.log_with_logger(logger)
        self.configured = True

    def __getstate__(self):
        return {
            "queue": self.queue,
            "name": self.name
        }

    def __setstate__(self, state):
        self.queue = state['queue']
        self.name = state['name']
        self.configured = False
        if multiprocessing.current_process().name == "MainProcess":
            return
        self.add_handler()


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

    def run(self):
        while not self.halting:
            try:
                message = self.message_queue.get(True, 2)
                self.handler(*message)
            except Exception:
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

    def __call__(self, *message):
        self.send(*message)

    def send(self, *message):
        self.queue.put(message)


def humanize_class_name(name):
    parts = []
    i = 0
    last = 0
    while i < len(name):
        c = name[i]
        if c.isupper() and i != last:
            if i + 1 < len(name):
                if name[i + 1].islower():
                    part = name[last:i]
                    parts.append(part)
                    last = i
        i += 1
    parts.append(name[last:i])
    return ' '.join(parts)


class LoggingMixin(object):
    logger_state = None
    print_fn = printer
    debug_print_fn = debug_printer
    error_print_fn = printer
    warn_print_fn = warnings.warn

    _debug_enabled = None

    @classmethod
    def log_with_logger(cls, logger):
        cls.logger_state = logger
        cls.print_fn = logger.info
        cls.debug_print_fn = logger.debug
        cls.error_print_fn = logger.error
        cls.warn_print_fn = logger.warning

    def instance_log_with_logger(self, logger):
        self.logger_state = logger
        self.print_fn = logger.info
        self.debug_print_fn = logger.debug
        self.error_print_fn = logger.error
        self.warn_print_fn = logger.warning

    @classmethod
    def log_to_stdout(cls):
        cls.logger_state = None
        cls.print_fn = printer
        cls.debug_print_fn = debug_printer
        cls.error_print_fn = printer
        cls.warn_print_fn = warnings.warn

    def log(self, *message):
        self.print_fn(u', '.join(map(ensure_text, message)), stacklevel=2)

    def debug(self, *message):
        self.debug_print_fn(u', '.join(map(ensure_text, message)), stacklevel=2)

    def error(self, *message, **kwargs):
        exception = kwargs.get("exception")
        self.error_print_fn(u', '.join(
            map(ensure_text, message)), stacklevel=2)
        if exception is not None:
            self.error_print_fn(traceback.format_exc())

    def warn(self, *message, **kwargs):
        self.warn_print_fn(u', '.join(map(ensure_text, message)), stacklevel=2)

    def ipc_logger(self, handler=None):
        return IPCLoggingManager()

    def in_debug_mode(self):
        if self._debug_enabled is None:
            logger_state = self.logger_state
            if logger_state is not None:
                self._debug_enabled = logger_state.isEnabledFor("DEBUG")
        return bool(self._debug_enabled)


class TaskBase(LoggingMixin):
    """A base class for a discrete, named step in a pipeline that
    executes in sequence.

    Attributes
    ----------
    debug_print_fn : Callable
        The function called to print debug messages
    display_fields : bool
        Whether to display fields at the start of execution
    end_time : datetime.datetime
        The time when the task ended
    error_print_fn : Callable
        The function called to print error messages
    logger_state : logging.Logger
        The Logger bound to this task
    print_fn : Callable
        The function called to print status messages
    start_time : datetime.datetime
        The time when the task began
    status : str
        The state of the executing task
    """

    status = "new"

    display_fields = True

    _display_name = None

    @property
    def display_name(self):
        if self._display_name is None:
            return humanize_class_name(self.__class__.__name__)
        else:
            return self._display_name

    def in_debug_mode(self):
        if self._debug_enabled is None:
            logger_state = self.logger_state
            if logger_state is not None:
                self._debug_enabled = logger_state.isEnabledFor(logging.DEBUG)
        return bool(self._debug_enabled)

    def _format_fields(self):
        if self.display_fields:
            return '\n' + pprint.pformat(
                {k: v for k, v in self.__dict__.items()
                 if not (k.startswith("_") or v is None)})
        else:
            return ''

    def display_header(self):
        display_version(self.log)

    def try_set_process_name(self, name=None):
        """
        This helper method may be used to try to change a process's name
        in order to make discriminating which role a particular process is
        fulfilling. This uses a third-party utility library that may not behave
        the same way on all platforms, and therefore this is done for convenience
        only.

        Parameters
        ----------
        name : str, optional
            A name to set. If not provided, will check the attribute ``process_name``
            for a non-null value, or else have no effect.
        """
        if name is None:
            name = getattr(self, 'process_name', None)
        if name is None:
            return
        _name_process(name)

    def _begin(self, verbose=True, *args, **kwargs):
        self.on_begin()
        self.start_time = datetime.now()
        self.status = "started"
        if verbose:
            self.log(
                "Begin %s%s" % (
                    self.display_name,
                    self._format_fields()))

    def _end(self, verbose=True, *args, **kwargs):
        self.on_end()
        self.end_time = datetime.now()
        if verbose:
            self.log("End %s" % self.display_name)
            self.log(self.summarize())

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def summarize(self):
        chunks = [
            "Started at %s." % self.start_time,
            "Ended at %s." % self.end_time,
            "Total time elapsed: %s" % (self.end_time - self.start_time),
            "%s completed successfully." % self.__class__.__name__ if self.status == 'completed' else
            "%s failed with error message %r" % (self.__class__.__name__, self.status),
            ''
        ]
        return '\n'.join(chunks)

    def start(self, *args, **kwargs):
        self._begin(*args, **kwargs)
        try:
            out = self.run()
        except (KeyboardInterrupt) as e:
            logger.exception("An error occurred: %r", e, exc_info=e)
            self.status = e
            out = e
            raise e
        else:
            self.status = 'completed'
        self._end(*args, **kwargs)
        return out

    def interact(self, **kwargs):
        from IPython.terminal.embed import InteractiveShellEmbed, load_default_config
        import sys
        config = kwargs.get('config')
        header = kwargs.pop('header', u'')
        compile_flags = kwargs.pop('compile_flags', None)
        if config is None:
            config = load_default_config()
            config.InteractiveShellEmbed = config.TerminalInteractiveShell
            kwargs['config'] = config
        frame = sys._getframe(1)
        shell = InteractiveShellEmbed.instance(
            _init_location_id='%s:%s' % (
                frame.f_code.co_filename, frame.f_lineno), **kwargs)
        shell(header=header, stack_depth=2, compile_flags=compile_flags,
              _call_location_id='%s:%s' % (frame.f_code.co_filename, frame.f_lineno))
        InteractiveShellEmbed.clear_instance()


log_handle = TaskBase()


class TaskExecutionSequence(TaskBase, Generic[T]):
    """A task unit that executes in a separate thread or process."""

    def __call__(self) -> T:
        result = None
        try:
            if self._running_in_process:
                self.log("%s running on PID %r" % (self, multiprocessing.current_process().pid))
            if os.getenv("GLYCRESOFTPROFILING"):
                import cProfile
                profiler = cProfile.Profile()
                result = profiler.runcall(self.run, standalone_mode=False)
                profiler.dump_stats('glycresoft_performance.profile')
            else:
                result = self.run()
            self.debug("%r Done" % self)
        except Exception as err:
            self.error("An error occurred while executing %s" %
                       self, exception=err)
            result = None
            self.set_error_occurred()
            try:
                self.done_event.set()
            except AttributeError:
                pass
        finally:
            return result

    def run(self) -> T:
        raise NotImplementedError()

    def _get_repr_details(self):
        return ''

    _thread: Optional[Union[threading.Thread, multiprocessing.Process]] = None
    _running_in_process: bool = False
    _error_flag: Optional[threading.Event] = None

    def error_occurred(self) -> bool:
        if self._error_flag is None:
            return False
        else:
            return self._error_flag.is_set()

    def set_error_occurred(self):
        if self._error_flag is None:
            return False
        else:
            return self._error_flag.set()

    def __repr__(self):
        template = "{self.__class__.__name__}({details})"
        return template.format(self=self, details=self._get_repr_details())

    def _make_event(self, provider=None) -> Union[threading.Event, multiprocessing.Event]:
        if provider is None:
            provider = threading
        return provider.Event()

    def _name_for_execution_sequence(self):
        return ("%s-%r" % (self.__class__.__name__, id(self)))

    def start(self, process: bool=False, daemon: bool=False):
        if self._thread is not None:
            return self._thread
        if process:
            self._running_in_process = True
            self._error_flag = self._make_event(multiprocessing)
            t = multiprocessing.Process(
                target=self, name=self._name_for_execution_sequence())
            if daemon:
                t.daemon = daemon
        else:
            self._error_flag = self._make_event(threading)
            t = threading.Thread(
                target=self, name=self._name_for_execution_sequence())
            if daemon:
                t.daemon = daemon
        t.start()
        self._thread = t
        return t

    def join(self, timeout: Optional[float]=None) -> bool:
        if self.error_occurred():
            return True
        try:
            return self._thread.join(timeout)
        except KeyboardInterrupt:
            self.set_error_occurred()
            return True

    def is_alive(self):
        if self.error_occurred():
            return False
        return self._thread.is_alive()

    def stop(self):
        if self.is_alive():
            self.set_error_occurred()

    def kill_process(self):
        if self._running_in_process:
            if self.is_alive():
                self._thread.terminate()
        else:
            self.log("Cannot kill a process running in a thread")


class Pipeline(TaskExecutionSequence):
    tasks: List[TaskExecutionSequence]
    error_polling_rate: float

    def __init__(self, tasks, error_polling_rate=1.0):
        self.tasks = tasks
        self.error_polling_rate = error_polling_rate

    def start(self, *args, **kwargs):
        for task in self:
            task.start(*args, **kwargs)

    def join(self, timeout: Optional[float]=None):
        if timeout is not None:
            for task in self:
                task.join(timeout)
        else:
            timeout = self.error_polling_rate
            while True:
                has_error = self.error_occurred()
                if has_error:
                    self.log("... Detected an error flag. Stopping!")
                    self.stop()
                    break
                alive = 0
                for task in self:
                    task.join(0.01)
                    is_alive = task.is_alive()
                    alive += is_alive
                if alive == 0:
                    break
                time.sleep(timeout)

    def is_alive(self):
        alive = 0
        for task in self:
            alive += task.is_alive()
        return alive

    def error_occurred(self) -> int:
        errors = 0
        for task in self.tasks:
            errors += task.error_occurred()
        return errors

    def stop(self):
        for task in self.tasks:
            task.stop()

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i: Union[int, slice]):
        return self.tasks[i]

    def add(self, task):
        self.tasks.append(task)
        return self


class SinkTask(TaskExecutionSequence):
    def __init__(self, in_queue, in_done_event):
        self.in_queue = in_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def handle_item(self, task):
        pass

    def process(self):
        has_work = True
        while has_work and not self.error_occurred():
            try:
                item = self.in_queue.get(True, 10)
                self.handle_item(item)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


def make_shared_memory_manager():
    manager = SyncManager()
    manager.start(_name_process, ("glycresoft-shm", ))
    return manager


def _name_process(name):
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except (ImportError, AttributeError):
        pass


def elapsed(seconds):
    '''Convert a second count into a human readable duration

    Parameters
    ----------
    seconds : :class:`int`
        The number of seconds elapsed

    Returns
    -------
    :class:`str` :
        A formatted, comma separated list of units of duration in days, hours, minutes, and seconds
    '''
    periods = [
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1)
    ]

    tokens = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            tokens.append("%s %s%s" % (period_value, period_name, has_s))

    return ", ".join(tokens)

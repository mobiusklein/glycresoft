from __future__ import print_function
import logging
import pprint
import traceback
import multiprocessing
import threading
import six
from multiprocessing.managers import SyncManager
from datetime import datetime

try:
    from Queue import Empty
except ImportError:
    from queue import Empty

from glycan_profiling.version import version


logger = logging.getLogger("glycan_profiling.task")


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
            self.error_print_fn(traceback.format_exc())

    def ipc_logger(self, handler=None):
        if handler is None:
            def _default_closure_handler(message):
                self.log(message)
            handler = _default_closure_handler
        return MessageSpooler(handler)


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

    _debug_enabled = None
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
                self._debug_enabled = logger_state.isEnabledFor("DEBUG")
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
        """This helper method may be used to try to change a process's name
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


class MultiEvent(object):
    def __init__(self, events):
        self.events = list(events)

    def set(self):
        for event in self.events:
            event.set()

    def is_set(self):
        for event in self.events:
            result = event.is_set()
            if not result:
                return result
        return True

    def wait(self, *args, **kwargs):
        result = True
        for event in self.events:
            result &= event.wait(*args, **kwargs)
        return result

    def clear(self):
        for event in self.events:
            event.clear()


class MultiLock(object):
    def __init__(self, locks):
        self.locks = list(locks)

    def acquire(self):
        for lock in self.locks:
            lock.acquire()

    def release(self):
        for lock in self.locks:
            lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


class TaskExecutionSequence(TaskBase):
    """A task unit that executes in a separate thread or process.
    """

    def __call__(self):
        result = None
        try:
            if self._running_in_process:
                self.log("%s running on PID %r" % (self, multiprocessing.current_process().pid))
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

    def run(self):
        raise NotImplementedError()

    def _get_repr_details(self):
        return ''

    _thread = None
    _running_in_process = False
    _error_flag = None

    def error_occurred(self):
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

    def _make_event(self, provider=None):
        if provider is None:
            provider = threading
        return provider.Event()

    def _name_for_execution_sequence(self):
        return ("%s-%r" % (self.__class__.__name__, id(self)))

    def start(self, process=False, daemon=False):
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

    def join(self, timeout=None):
        if self.error_occurred():
            return True
        return self._thread.join(timeout)

    def is_alive(self):
        if self.error_occurred():
            return False
        return self._thread.is_alive()

    def stop(self):
        if self.is_alive():
            self.set_error_occurred()


class Pipeline(TaskExecutionSequence):
    def __init__(self, tasks):
        self.tasks = tasks

    def start(self, *args, **kwargs):
        for task in self:
            task.start(*args, **kwargs)

    def join(self, timeout=None):
        if timeout is not None:
            for task in self:
                task.join(timeout)
        else:
            timeout = max(60 // len(self), 2)
            while True:
                has_error = self.error_occurred()
                if has_error:
                    for task in self:
                        task.stop()
                alive = 0
                for task in self:
                    task.join(timeout)
                    is_alive = task.is_alive()
                    alive += is_alive
                if alive == 0:
                    break

    def is_alive(self):
        alive = 0
        for task in self:
            alive += task.is_alive()
        return alive

    def error_occurred(self):
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

    def __getitem__(self, i):
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

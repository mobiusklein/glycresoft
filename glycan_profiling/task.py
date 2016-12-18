from __future__ import print_function
import logging
import pprint
import traceback
from datetime import datetime
import multiprocessing
import threading

logger = logging.getLogger("glycan_profiling.task")


def printer(obj, message):
    print(datetime.now().isoformat(' ') + ' ' + str(message))


def debug_printer(obj, message):
    if obj.debug:
        print("DEBUG:" + datetime.now().isoformat(' ') + ' ' + str(message))


class MessageSpooler(object):
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
                self.handler(message)
            except:
                continue

    def stop(self):
        self.halting = True
        self.thread.join()

    def sender(self):
        return MessageSender(self.queu)


class MessageSender(object):
    def __init__(self, queue):
        self.queue = queue

    def __call__(self, message):
        self.send(message)

    def send(self, message):
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


class TaskBase(object):
    status = "new"

    debug = False

    print_fn = printer
    debug_print_fn = debug_printer
    error_print_fn = printer
    display_fields = True

    _display_name = None

    @property
    def display_name(self):
        if self._display_name is None:
            return humanize_class_name(self.__class__.__name__)
        else:
            return self._display_name

    @classmethod
    def log_with_logger(cls, logger):
        cls.print_fn = logger.info
        cls.debug_print_fn = logger.debug
        cls.error_print_fn = logger.error

    def _format_fields(self):
        if self.display_fields:
            return '\n' + pprint.pformat(
                {k: v for k, v in self.__dict__.items()
                 if not k.startswith("_")})
        else:
            return ''

    def log(self, message):
        self.print_fn(str(message))

    def error(self, message, exception=None):
        self.error_print_fn(str(message))
        if exception is not None:
            self.error_print_fn(traceback.format_exc(exception))

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
        except (KeyboardInterrupt), e:
            logger.exception("An error occurred: %r", e, exc_info=e)
            self.status = e
            out = e
            raise e
        else:
            self.status = 'completed'
        self._end(*args, **kwargs)
        return out


log_handle = TaskBase()

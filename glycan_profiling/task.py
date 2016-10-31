from __future__ import print_function
import logging
import pprint
import traceback
from datetime import datetime

logger = logging.getLogger("glycan_profiling.task")


def printer(obj, message):
    print(datetime.now().isoformat(' ') + ' ' + str(message))


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

    print_fn = printer
    error_print_fn = printer
    display_fields = False

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
        self.start_time = datetime.now()
        self.status = "started"
        if verbose:
            self.log(
                "Begin %s%s" % (
                    self.display_name,
                    self._format_fields()))
        self.on_begin()

    def _end(self, verbose=True, *args, **kwargs):
        self.end_time = datetime.now()
        if verbose:
            self.log("End %s" % self.display_name)
            self.log(self.summarize())
        self.on_end()

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

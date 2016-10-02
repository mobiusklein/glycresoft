import logging
import pprint
from datetime import datetime

logger = logging.getLogger("glycan_profiling.task")


class TaskBase(object):
    status = "new"

    def log(self, message):
        print(datetime.now().isoformat(' ') + ' ' + str(message))

    def error(self, message):
        print(datetime.now().isoformat(' ') + ' ' + str(message))

    def _begin(self, verbose=True, *args, **kwargs):
        self.start_time = datetime.now()
        self.status = "started"
        if verbose:
            logger.info("Begin %s\n%s\n", self.__class__.__name__, pprint.pformat(
                {k: v for k, v in self.__dict__.items() if not k.startswith("_")}))
        self.on_begin()

    def _end(self, verbose=True, *args, **kwargs):
        self.end_time = datetime.now()
        if verbose:
            logger.info("End %s", self.__class__.__name__)
            logger.info(self.summarize())
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
            "%s failed with error message %r" % (self.__class__.__name__, self.status)
        ]
        return '\n'.join(chunks)

    def start(self, *args, **kwargs):
        self._begin(*args, **kwargs)
        try:
            out = self.run()
        except (Exception), e:
            logger.exception("An error occurred: %r", e, exc_info=e)
            self.status = e
        else:
            self.status = 'completed'
        return out

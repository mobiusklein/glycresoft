from glypy.utils import Enum


class SentinelToken(object):
    """An object to hold opaque identity information regarding a worker process,
    to be used to signal that a worker process has received a signal

    Attributes
    ----------
    token : object
        The opaque identity
    """

    def __init__(self, token):
        self.token = token

    def __hash__(self):
        return hash(self.token)

    def __eq__(self, other):
        return self.token == other.token

    def __repr__(self):
        return "{self.__class__.__name__}({self.token})".format(self=self)


class ProcessDispatcherState(Enum):
    start = 1
    spawning = 2
    running = 3
    running_local_workers_live = 4
    running_local_workers_dead = 5
    terminating = 6
    terminating_workers_live = 7
    done = 8

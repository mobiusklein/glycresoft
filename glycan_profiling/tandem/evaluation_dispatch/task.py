import os
from collections import deque

from glycan_profiling.task import TaskBase


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))


class StructureSpectrumSpecificationBuilder(object):
    """Base class for building structure hit by spectrum specification
    """

    def build_work_order(self, hit_id, hit_map, scan_hit_type_map, hit_to_scan):
        """Packs several task-defining data structures into a simple to unpack payload for
        sending over IPC to worker processes.

        Parameters
        ----------
        hit_id : int
            The id number of a hit structure
        hit_map : dict
            Maps hit_id to hit structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match

        Returns
        -------
        tuple
            Packaged message payload
        """
        return (hit_map[hit_id],
                [(s, scan_hit_type_map[s, hit_id])
                 for s in hit_to_scan[hit_id]])


class TaskSourceBase(StructureSpectrumSpecificationBuilder, TaskBase):
    """A base class for building a stream of work items through
    :class:`StructureSpectrumSpecificationBuilder`.
    """

    batch_size = 10000

    def add(self, item):
        """Add ``item`` to the work stream

        Parameters
        ----------
        item : object
            The work item to deal
        """
        raise NotImplementedError()

    def join(self):
        """Checkpoint that may halt the stream generation.
        """
        return

    def feed(self, hit_map, hit_to_scan, scan_hit_type_map):
        """Push tasks onto the input queue feeding the worker
        processes.

        Parameters
        ----------
        hit_map : dict
            Maps hit id to structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match
        """
        i = 0
        n = len(hit_to_scan)
        seen = dict()
        for hit_id, scan_ids in hit_to_scan.items():
            i += 1
            hit = hit_map[hit_id]
            # This sanity checking is likely unnecessary, and is a hold-over from
            # debugging redundancy in the result queue. For the moment, it is retained
            # to catch "new" bugs.
            # If a hit structure's id doesn't match the id it was looked up with, something
            # may be wrong with the upstream process. Log this event.
            if hit.id != hit_id:
                self.log("Hit %r doesn't match its id %r" % (hit, hit_id))
                if hit_to_scan[hit.id] != scan_ids:
                    self.log("Mismatch leads to different scans! (%d, %d)" % (
                        len(scan_ids), len(hit_to_scan[hit.id])))
            # If a hit structure has been seen multiple times independent of whether or
            # not the expected hit id matches, something may be wrong in the upstream process.
            # Log this event.
            if hit.id in seen:
                self.log("Hit %r already dealt under hit_id %r, now again at %r" % (
                    hit, seen[hit.id], hit_id))
                raise ValueError(
                    "Hit %r already dealt under hit_id %r, now again at %r" % (
                        hit, seen[hit.id], hit_id))
            seen[hit.id] = hit_id
            if i % self.batch_size == 0 and i:
                self.join()
            try:
                work_order = self.build_work_order(hit_id, hit_map, scan_hit_type_map, hit_to_scan)
                # if debug_mode:
                #     self.log("...... Matching %s against %r" % work_order)
                self.add(work_order)
                # Set a long progress update interval because the feeding step is less
                # important than the processing step. Additionally, as the two threads
                # run concurrently, the feeding thread can log a short interval before
                # the entire process has formally logged that it has started.
                if i % 10000 == 0:
                    self.log("...... Dealt %d work items (%0.2f%% Complete)" % (i, i * 100.0 / n))
            except Exception as e:
                self.log("An exception occurred while feeding %r and %d scan ids: %r" % (hit_id, len(scan_ids), e))
        self.log("...... Finished dealing %d work items" % (i,))
        self.join()
        return

    def feed_groups(self, hit_map, hit_to_scan, scan_hit_type_map, hit_to_group):
        """Push task groups onto the input queue feeding the worker
        processes.

        Parameters
        ----------
        hit_map : dict
            Maps hit id to structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match
        hit_to_group: dict
            Maps group id to the set of hit ids which are
        """
        i = 0
        j = 0
        n = len(hit_to_group)
        seen = dict()
        for group_key, hit_keys in hit_to_group.items():
            hit_group = {
                "hits": {},
                "work_orders": {}
            }
            i += 1
            for hit_id in hit_keys:
                j += 1
                scan_ids = hit_to_scan[hit_id]
                hit = hit_map[hit_id]
                hit_group['hits'][hit_id] = hit
                # This sanity checking is likely unnecessary, and is a hold-over from
                # debugging redundancy in the result queue. For the moment, it is retained
                # to catch "new" bugs.
                # If a hit structure's id doesn't match the id it was looked up with, something
                # may be wrong with the upstream process. Log this event.
                if hit.id != hit_id:
                    self.log("Hit %r doesn't match its id %r" % (hit, hit_id))
                    if hit_to_scan[hit.id] != scan_ids:
                        self.log("Mismatch leads to different scans! (%d, %d)" % (
                            len(scan_ids), len(hit_to_scan[hit.id])))
                # If a hit structure has been seen multiple times independent of whether or
                # not the expected hit id matches, something may be wrong in the upstream process.
                # Log this event.
                if hit.id in seen:
                    self.log("Hit %r already dealt under hit_id %r, now again at %r in group %r" % (
                        hit, seen[hit.id], hit_id, group_key))
                    raise ValueError(
                        "Hit %r already dealt under hit_id %r, now again at %r" % (
                            hit, seen[hit.id], hit_id))
                seen[hit.id] = (hit_id, group_key)
                work_order = self.build_work_order(
                    hit_id, hit_map, scan_hit_type_map, hit_to_scan)
                hit_group['work_orders'][hit_id] = work_order
            self.add(hit_group)
            if i % self.batch_size == 0 and i:
                self.join()
        self.log("...... Finished dealing %d work items" % (i,))
        self.join()
        return

    def __call__(self, hit_map, hit_to_scan, scan_hit_type_map):
        return self.feed(hit_map, hit_to_scan, scan_hit_type_map)


class TaskDeque(TaskSourceBase):
    """Generate an on-memory buffer of work items

    Attributes
    ----------
    queue : :class:`~.deque`
        The in-memory work queue
    """

    def __init__(self):
        self.queue = deque()

    def add(self, item):
        self.queue.append(item)

    def pop(self):
        return self.queue.popleft()

    def __iter__(self):
        return iter(self.queue)


class TaskQueueFeeder(TaskSourceBase):
    def __init__(self, input_queue, done_event):
        self.input_queue = input_queue
        self.done_event = done_event

    def add(self, item):
        self.input_queue.put(item)

    def join(self):
        return self.input_queue.join()

    def feed(self, hit_map, hit_to_scan, scan_hit_type_map):
        """Push tasks onto the input queue feeding the worker
        processes.

        Parameters
        ----------
        hit_map : dict
            Maps hit id to structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match
        """
        super(TaskQueueFeeder, self).feed(hit_map, hit_to_scan, scan_hit_type_map)
        self.done_event.set()
        return

    def __call__(self, hit_map, hit_to_scan, scan_hit_type_map):
        return self.feed(hit_map, hit_to_scan, scan_hit_type_map)

from typing import Optional, Union
from weakref import WeakValueDictionary

from ms_deisotope.data_source import ProcessedScan, PrecursorInformation, ActivationInformation, Scan
from ms_deisotope.data_source import ProcessedRandomAccessScanSource


class ScanStub(object):
    """A stub for holding precursor information and
    giving a Scan-like interface for accessing just that
    information. Provides a serialized-like interface
    which clients can use to load the real scan.

    Attributes
    ----------
    id : str
        The scan ID for the proxied scan
    precursor_information : :class:`~.PrecursorInformation`
        The information describing the relevant
        metadata for scheduling when and where this
        scan should be processed, where actual loading
        will occur.
    source : :class:`~.RandomAccessScanSource`
        A resource to use to load scans with by scan id.
    """

    id: str
    precursor_information: PrecursorInformation
    source: Optional[ProcessedRandomAccessScanSource]

    def __init__(self, precursor_information: PrecursorInformation, source: ProcessedRandomAccessScanSource):
        self.id = precursor_information.product_scan_id
        self.precursor_information = precursor_information
        self.source = source

    def detatch(self) -> 'ScanStub':
        self.source = None
        self.precursor_information.source = None
        return self

    def unbind(self):
        return self.detatch()

    def bind(self, source):
        self.source = source
        self.precursor_information.source = source
        return self

    @property
    def scan_id(self):
        return self.id

    def convert(self, *args, **kwargs) -> Union[ProcessedScan, Scan]:
        try:
            return self.source.get_scan_by_id(self.id)
        except AttributeError:
            raise KeyError(self.id)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.precursor_information.neutral_mass}, {self.precursor_information})"
        return template.format(self=self)


class ScanWrapperBase(object):
    __slots__ = []

    scan: Union[ProcessedScan, Scan]

    @property
    def scan_id(self):
        self.requires_scan()
        return self.scan.scan_id

    @property
    def precursor_ion_mass(self):
        self.requires_scan()
        neutral_mass = self.scan.precursor_information.extracted_neutral_mass
        if neutral_mass == 0:
            neutral_mass = self.scan.precursor_information.neutral_mass
        return neutral_mass

    @property
    def scan_time(self):
        self.requires_scan()
        return self.scan.scan_time

    @property
    def precursor_information(self):
        self.requires_scan()
        return self.scan.precursor_information

    def requires_scan(self):
        if self.scan is None:
            raise ValueError("%s is detatched from Scan" % (self.__class__.__name__))

    @property
    def ms_level(self):
        self.requires_scan()
        return self.scan.ms_level

    @property
    def index(self):
        self.requires_scan()
        return self.scan.index

    @property
    def activation(self):
        self.requires_scan()
        return self.scan.activation

    @property
    def deconvoluted_peak_set(self):
        self.requires_scan()
        return self.scan.deconvoluted_peak_set

    @property
    def peak_set(self):
        self.requires_scan()
        return self.scan.peak_set

    @property
    def arrays(self):
        self.requires_scan()
        return self.scan.arrays

    @property
    def annotations(self):
        self.requires_scan()
        return self.scan.annotations


class ScanInformation(object):
    """A carrier of scan-level metadata that does not include peak data to reduce space
    consumption.
    """

    __slots__ = ("id", "index", "scan_time", "ms_level",
                 "precursor_information", "activation",
                 "title", "__weakref__")

    id: str
    index: int
    scan_time: float
    ms_level: int
    precursor_information: PrecursorInformation
    activation: ActivationInformation
    title: str

    def __init__(self, scan_id, index, scan_time, ms_level, precursor_information, activation=None, title=None):
        self.id = scan_id
        self.index = index
        self.scan_time = scan_time
        self.ms_level = ms_level
        self.precursor_information = precursor_information
        self.activation = activation
        self.title = title

    def __reduce__(self):
        return self.__class__, (self.id, self.index, self.scan_time, self.ms_level,
                                self.precursor_information, self.activation, self.title)

    @property
    def scan_id(self):
        return self.id

    @classmethod
    def from_scan(cls, scan):
        return cls(
            scan.scan_id, scan.index, scan.scan_time, scan.ms_level,
            scan.precursor_information, scan.activation, scan.title)

    def __repr__(self):
        template = ("{self.__class__.__name__}({self.id}, {self.index}, {self.scan_time}, "
                    "{self.ms_level}, {self.precursor_information}, {self.activation}, {self.title})")
        return template.format(self=self)


class ScanInformationLoader(object):
    scan_loader: ProcessedRandomAccessScanSource

    def __init__(self, scan_loader):
        self.scan_loader = scan_loader
        self.cache = WeakValueDictionary()

    def get_scan_by_id(self, scan_id):
        try:
            return self.cache[scan_id]
        except KeyError:
            scan = self.scan_loader.get_scan_by_id(scan_id)
            info = ScanInformation.from_scan(scan)
            self.cache[scan_id] = info
            return info


def top_n_peaks(scan: ProcessedScan, n: int=300) -> ProcessedScan:
    scan = scan.clone(deep=True)
    peaks = scan.deconvoluted_peak_set.__class__(sorted(
        scan.deconvoluted_peak_set, key=lambda x: x.intensity, reverse=True)[:n])
    peaks.reindex()
    scan.deconvoluted_peak_set = peaks
    return scan

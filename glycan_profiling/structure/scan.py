from ms_deisotope.utils import Base


class ScanStub(object):
    """A stub for holding precursor information and
    giving a Scan-like interface for accessing just that
    information. Provides a serialized-like interface
    which clients can use to load the real scan.

    Attributes
    ----------
    id : str
        The scan ID for the proxied scan
    precursor_information : PrecursorInformation
        The information describing the relevant
        metadata for scheduling when and where this
        scan should be processed, where actual loading
        will occur.
    bind : MzMLLoader
        A resource to use to load scans with by scan id.
    """
    def __init__(self, precursor_information, bind):
        self.id = precursor_information.product_scan_id
        self.precursor_information = precursor_information
        self.bind = bind

    @property
    def scan_id(self):
        return self.id

    def convert(self, *args, **kwargs):
        try:
            return self.bind.get_scan_by_id(self.id)
        except AttributeError:
            raise KeyError(self.id)


class ScanWrapperBase(object):
    __slots__ = []

    @property
    def scan_id(self):
        self.requires_scan()
        return self.scan.scan_id

    @property
    def precursor_ion_mass(self):
        self.requires_scan()
        neutral_mass = self.scan.precursor_information.extracted_neutral_mass
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


class ScanInformation(Base):
    def __init__(self, scan_id, index, scan_time, ms_level, precursor_information):
        self.id = scan_id
        self.index = index
        self.scan_time = scan_time
        self.ms_level = ms_level
        self.precursor_information = precursor_information

    @property
    def scan_id(self):
        return self.id

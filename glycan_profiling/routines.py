from .serialize import DatabaseScanDeserializer
from .scan_cache import NullScanCacheHandler
from .trace import IncludeUnmatchedTracer, ChromatogramFilter
from .chromatogram_tree import ChromatogramOverlapSmoother

from .tandem import disk_backed_database
from .tandem.glycopeptide import glycopeptide_matcher
from .tandem.glycopeptide import BinomialSpectrumMatcher, TargetDecoyAnalyzer

try:
    basestring
except:
    basestring = (str, bytes)


def load_and_marshal_chromatograms(deserializer, minimum_mass=1000., n_peaks=3):
    if isinstance(deserializer, basestring):
        deserializer = DatabaseScanDeserializer(deserializer)
    accumulated_peaks = deserializer.ms1_peaks_above(minimum_mass)
    accumulate = [x[:2] for x in accumulated_peaks]
    t = IncludeUnmatchedTracer(deserializer, [], cache_handler_type=NullScanCacheHandler)
    t.unmatched.extend(accumulate)
    chroma = t.chromatograms(truncate=False, minimum_mass=minimum_mass)
    pf = ChromatogramFilter.process(chroma, n_peaks=n_peaks)
    pf = ChromatogramFilter(ChromatogramOverlapSmoother(pf))
    result = {
        "chromatograms": pf,
        "deserializer": deserializer,
        "accumulated_peaks": accumulated_peaks
    }
    return result


def flatten(iterable):
    return [a for b in iterable for a in b]

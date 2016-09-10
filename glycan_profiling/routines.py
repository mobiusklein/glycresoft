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


def glycopeptide_database_search(deserializer, chromatograms, structure_database, structure_hypothesis=None):
    if isinstance(deserializer, basestring):
        deserializer = DatabaseScanDeserializer(deserializer)
    if isinstance(structure_database, basestring):
        structure_database = disk_backed_database.DiskBackedStructureDatabase(
            structure_database, structure_hypothesis, 5)
    target_hits = []
    decoy_hits = []
    for f in sorted(chromatograms, key=lambda x: x.neutral_mass, reverse=True):
        if f.has_msms:
            print(f, len(f.has_msms))
            print("Extracting MS/MS")
            cluster = [x.product.convert() for x in deserializer.msms_for(
                f.neutral_mass, 1e-5, f.start_time, f.end_time)]
            print(len(cluster))
            tce = glycopeptide_matcher.GlycopeptideMatcher(cluster[:], BinomialSpectrumMatcher, structure_database)
            dce = glycopeptide_matcher.DecoyGlycopeptideMatcher(cluster[:], BinomialSpectrumMatcher, structure_database)
            print("Scoring Targets")
            target_hits.extend(tce.score_all(error_tolerance=2e-5, simplify=True))
            print(len(target_hits))
            print("Scoring Decoys")
            decoy_hits.extend(dce.score_all(error_tolerance=2e-5, simplify=True))
    TargetDecoyAnalyzer(flatten(target_hits), flatten(decoy_hits), with_pit=False).q_values()
    return target_hits

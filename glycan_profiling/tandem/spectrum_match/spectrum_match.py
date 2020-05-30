import warnings
import struct

from glypy.utils import Enum, make_struct

from ms_deisotope import DeconvolutedPeakSet, isotopic_shift
from ms_deisotope.data_source.metadata import activation

from glycan_profiling.structure import (
    ScanWrapperBase)

from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.tandem.ref import TargetReference, SpectrumReference


neutron_offset = isotopic_shift()

LowCID = activation.dissociation_methods['low-energy collision-induced dissociation']


class SpectrumMatchBase(ScanWrapperBase):
    """A base class for spectrum matches, a scored pairing between a structure
    and tandem mass spectrum.

    Attributes
    ----------
    scan: :class:`ms_deisotope.ProcessedScan`
        The processed MSn spectrum to match
    target: :class:`object`
        A structure that can be fragmented and scored against.
    mass_shift: :class:`~.MassShift`
        A mass shifting adduct that alters the precursor mass and optionally some
        of the fragment masses.

    """
    __slots__ = ['scan', 'target', "_mass_shift"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.target = target
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def mass_shift(self):
        """A mass shifting adduct that alters the precursor mass and optionally some
        of the fragment masses.

        Returns
        -------
        :class:`~.MassShift`
        """
        return self._mass_shift

    @mass_shift.setter
    def mass_shift(self, value):
        self._mass_shift = value

    @staticmethod
    def threshold_peaks(deconvoluted_peak_set, threshold_fn=lambda peak: True):
        """Filter a deconvoluted peak set by a predicate function.

        Parameters
        ----------
        deconvoluted_peak_set : :class:`ms_deisotope.DeconvolutedPeakSet`
            The deconvoluted peaks to filter
        threshold_fn : Callable
            The predicate function to use to decide whether or not to keep a peak.

        Returns
        -------
        :class:`ms_deisotope.DeconvolutedPeakSet`
        """
        deconvoluted_peak_set = DeconvolutedPeakSet([
            p for p in deconvoluted_peak_set
            if threshold_fn(p)
        ])
        deconvoluted_peak_set._reindex()
        return deconvoluted_peak_set

    def _theoretical_mass(self):
        return self.target.total_composition().mass

    def precursor_mass_accuracy(self, offset=0):
        """Calculate the precursor mass accuracy in PPM, accounting for neutron offset
        and mass shift.

        Parameters
        ----------
        offset : int, optional
            The number of neutron errors to account for (the default is 0).

        Returns
        -------
        float:
            The precursor mass accuracy in PPM.
        """
        observed = self.precursor_ion_mass
        theoretical = self._theoretical_mass() + (
            offset * neutron_offset) + self.mass_shift.mass
        return (observed - theoretical) / theoretical

    def determine_precursor_offset(self, probing_range=3, include_error=False):
        """Iteratively re-estimate what the actual offset to the precursor mass was.

        Parameters
        ----------
        probing_range : int, optional
            The range of neutron errors to account for. (the default is 3)
        include_error : bool, optional
            Whether or not to return both the offset and the estimated mass error in PPM.
            Defaults to False.

        Returns
        -------
        best_offset: int
            The best precursor offset.
        error: float
            The precursor mass error (in PPM) if `include_error` is :const:`True`
        """
        best_offset = 0
        best_error = float('inf')
        theoretical_mass_base = self._theoretical_mass() + self.mass_shift.mass
        observed = self.precursor_ion_mass
        for i in range(probing_range + 1):
            theoretical = theoretical_mass_base + i * neutron_offset
            error = abs((observed - theoretical) / theoretical)
            if error < best_error:
                best_error = error
                best_offset = i
        if include_error:
            return best_offset, best_error
        return best_offset

    def __reduce__(self):
        return self.__class__, (self.scan, self.target)

    def get_top_solutions(self):
        return [self]

    def __eq__(self, other):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        try:
            other_target_id = self.target.id
        except AttributeError:
            other_target_id = None
        return (self.scan == other.scan) and (self.target == other.target) and (
            target_id == other_target_id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        return hash((self.scan.id, self.target, target_id))

    def is_hcd(self):
        """Check whether the MSn spectrum was fragmented using a collisional dissociation
        mechanism or not.

        The result is cached on :attr:`scan`, so the interpretation does not need to be repeated.

        .. note:: If no activation information is present, the spectrum will be assumed to be HCD.

        Returns
        -------
        bool
        """
        annotations = self.annotations
        try:
            result = annotations['is_hcd']
        except KeyError:
            activation_info = self.scan.activation
            if activation_info is None:
                if self.scan.ms_level == 1:
                    result = False
                else:
                    warnings.warn("Activation information is missing. Assuming HCD")
                    result = True
            else:
                result = activation_info.has_dissociation_type(activation.HCD) or\
                    activation_info.has_dissociation_type(activation.CID) or\
                    activation_info.has_dissociation_type(LowCID) or\
                    activation_info.has_dissociation_type(activation.UnknownDissociation)
            annotations['is_hcd'] = result
        return result

    def is_exd(self):
        """Check if the scan was dissociated using an E*x*D method.

        This checks for ECD and ETD terms.

        This method caches its result in the scan's annotations.

        Returns
        -------
        bool

        See Also
        --------
        is_hcd
        """
        annotations = self.annotations
        try:
            result = annotations['is_exd']
        except KeyError:

            activation_info = self.scan.activation
            if activation_info is None:
                if self.scan.ms_level == 1:
                    result = False
                else:
                    warnings.warn("Activation information is missing. Assuming not ExD")
                    result = False
            else:
                result = activation_info.has_dissociation_type(activation.ETD) or\
                    activation_info.has_dissociation_type(activation.ECD)
            annotations['is_exd'] = result
        return result

    def mz_range(self):
        annotations = self.annotations
        try:
            result = annotations['mz_range']
        except KeyError:
            acquisition_info = self.scan.acquisition_information
            if acquisition_info is None:
                mz_range = (0, 1e6)
            else:
                lo = float('inf')
                hi = 0
                for event in acquisition_info:
                    for window in event:
                        lo = min(window.lower, lo)
                        hi = max(window.upper, hi)
                # No events/windows or an error
                if hi < lo:
                    mz_range = (0, 1e6)
                else:
                    mz_range = (lo, hi)
            annotations['mz_range'] = mz_range
            result = mz_range
        return result

    def get_auxiliary_data(self):
        return {}


class SpectrumMatcherBase(SpectrumMatchBase):
    __slots__ = ["spectrum", "_score"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.spectrum = scan.deconvoluted_peak_set
        self.target = target
        self._score = 0
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def score(self):
        """The aggregate spectrum match score.

        Returns
        -------
        float
        """
        return self._score

    def match(self, *args, **kwargs):
        """Match theoretical fragments against experimental peaks.
        """
        raise NotImplementedError()

    def calculate_score(self, *args, **kwargs):
        """Calculate a score given the fragment-peak matching.

        This method should populate :attr:`_score`.

        Returns
        -------
        float
        """
        raise NotImplementedError()

    def base_peak(self):
        """Find the base peak intensity of the spectrum, the
        most intense peak's intensity.

        Returns
        -------
        float

        """
        try:
            return max([peak.intensity for peak in self.spectrum])
        except ValueError:
            return 0

    @classmethod
    def evaluate(cls, scan, target, *args, **kwargs):
        """A high level method to construct a :class:`SpectrumMatcherBase`
        instance over a scan and target, call :meth:`match`, and
        :meth:`calculate_score`.

        Parameters
        ----------
        scan : :class:`~.ProcessedScan`
            The scan to match against.
        target : :class:`object`
            The structure to match.

        Returns
        -------
        :class:`SpectrumMatcherBase`
        """
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        inst = cls(scan, target, mass_shift=mass_shift)
        inst.match(*args, **kwargs)
        inst.calculate_score(*args, **kwargs)
        return inst

    def __getstate__(self):
        return (self.score,)

    def __setstate__(self, state):
        self.score = state[0]

    def __reduce__(self):
        return self.__class__, (self.scan, self.target,)

    @staticmethod
    def load_peaks(scan):
        try:
            return scan.convert(fitted=False, deconvoluted=True)
        except AttributeError:
            return scan

    def __repr__(self):
        return "{self.__class__.__name__}({self.scan_id}, {self.spectrum}, {self.target}, {self.score})".format(
            self=self)

    def plot(self, ax=None, **kwargs):
        """Plot the spectrum match, using the :class:`~.TidySpectrumMatchAnnotator`
        algorithm.

        Parameters
        ----------
        ax : :class:`matplotlib.Axis`, optional
            The axis to draw on. If not provided, a new figure
            and axis will be created.

        Returns
        -------
        :class:`~.TidySpectrumMatchAnnotator`
        """
        from glycan_profiling.plotting import spectral_annotation
        art = spectral_annotation.TidySpectrumMatchAnnotator(self, ax=ax)
        art.draw(**kwargs)
        return art


try:
    from glycan_profiling._c.tandem.tandem_scoring_helpers import base_peak
    SpectrumMatcherBase.base_peak = base_peak
except ImportError:
    pass


class SpectrumMatch(SpectrumMatchBase):
    """Represent a summarized spectrum match, which has been calculated already.

    Attributes
    ----------
    score: float
        The aggregate match score of the spectrum-structure pairing
    best_match: bool
        Whether or not this spectrum match is the best match for :attr:`scan`
    q_value: float
        The false discovery rate for the match
    id: int
        The unique identifier of the match

    """

    __slots__ = ['score', 'best_match', 'data_bundle', "q_value", 'id']

    def __init__(self, scan, target, score, best_match=False, data_bundle=None,
                 q_value=None, id=None, mass_shift=None):
        # if data_bundle is None:
        #     data_bundle = dict()
        super(SpectrumMatch, self).__init__(scan, target, mass_shift)

        self.score = score
        self.best_match = best_match
        self.data_bundle = data_bundle
        self.q_value = q_value
        self.id = id

    def is_multiscore(self):
        """Check whether this match has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return False

    def pack(self):
        """Package up the information of the match into the minimal information to
        describe the match.

        This method returns an opaque data carrier that can be reconstituted using
        :meth:`unpack`.

        Returns
        -------
        object
        """
        return (self.target.id, self.score, int(self.best_match), self.mass_shift.name)

    @classmethod
    def unpack(cls, data, spectrum, resolver, offset=0):
        """Reconstitute a :class:`SpectrumMatch` from packed
        data and a lookup table.

        Parameters
        ----------
        data : object
            The result of :meth:`pack`
        spectrum : :class:`~.ProcessedScan` or :class:`~.SpectrumReference`
            The spectrum that was matched.
        resolver : :class:`object`
            An object which can be used to resolve mass shifts and targets.
        offset : int, optional
            The offset into `data` to start reading from. Defaults to `0`.

        Returns
        -------
        match: :class:`SpectrumMatch`
            The reconstructed match
        offset: int
            The next offset to read from in `data`
        """
        i = offset
        target_id = int(data[i])
        score = float(data[i + 1])
        try:
            best_match = bool(int(data[i + 2]))
        except ValueError:
            best_match = bool(data[i + 2])
        mass_shift_name = data[i + 3]
        mass_shift = resolver.resolve_mass_shift(mass_shift_name)
        match = SpectrumMatch(
            spectrum,
            resolver.resolve_target(target_id),
            score,
            best_match,
            mass_shift=mass_shift)
        i += 4
        return match, i

    def clear_caches(self):
        """Clear caches on :attr:`target`.
        """
        try:
            self.target.clear_caches()
        except AttributeError:
            pass

    def __reduce__(self):
        return self.__class__, (self.scan, self.target, self.score, self.best_match,
                                self.data_bundle, self.q_value, self.id, self.mass_shift)

    def evaluate(self, scorer_type, *args, **kwargs):
        """Re-evaluate this spectrum-structure pair.

        Parameters
        ----------
        scorer_type : :class:`SpectrumMatcherBase`
            The matcher type to use.

        Returns
        -------
        :class:`SpectrumMatcherBase`
        """
        if isinstance(self.scan, SpectrumReference):
            raise TypeError("Cannot evaluate a spectrum reference")
        elif isinstance(self.target, TargetReference):
            raise TypeError("Cannot evaluate a target reference")
        return scorer_type.evaluate(self.scan, self.target, *args, **kwargs)

    def __repr__(self):
        return "%s(%s, %s, %0.4f, %r)" % (
            self.__class__.__name__,
            self.scan, self.target, self.score, self.mass_shift)

    @classmethod
    def from_match_solution(cls, match):
        """Create a :class:`SpectrumMatch` from another :class:`SpectrumMatcherBase`

        Parameters
        ----------
        match : :class:`SpectrumMatcherBase`
            The :class:`SpectrumMatcherBase` to convert

        Returns
        -------
        :class:`SpectrumMatch`
        """
        return cls(match.scan, match.target, match.score, mass_shift=match.mass_shift)

    def clone(self):
        """Create a shallow copy of this object

        Returns
        -------
        :class:`SpectrumMatch`
        """
        return self.__class__(
            self.scan, self.target, self.score, self.best_match, self.data_bundle,
            self.q_value, self.id, self.mass_shift)

    def get_auxiliary_data(self):
        return self.data_bundle


class ModelTreeNode(object):
    def __init__(self, model, children=None):
        if children is None:
            children = {}
        self.children = children
        self.model = model

    def get_model_node_for(self, scan, target, *args, **kwargs):
        for decider, model_node in self.children.items():
            if decider(scan, target, *args, **kwargs):
                return model_node.get_model_node_for(scan, target, *args, **kwargs)
        return self

    def evaluate(self, scan, target, *args, **kwargs):
        node = self.get_model_node_for(scan, target, *args, **kwargs)
        return node.model.evaluate(scan, target, *args, **kwargs)

    def __call__(self, scan, target, *args, **kwargs):
        node = self.get_model_node_for(scan, target, *args, **kwargs)
        return node.model(scan, target, *args, **kwargs)

    def load_peaks(self, scan):
        return self.model.load_peaks(scan)


class SpectrumMatchClassification(Enum):
    target_peptide_target_glycan = 0
    target_peptide_decoy_glycan = 1
    decoy_peptide_target_glycan = 2
    decoy_peptide_decoy_glycan = decoy_peptide_target_glycan | target_peptide_decoy_glycan


_ScoreSet = make_struct("ScoreSet", ['glycopeptide_score', 'peptide_score', 'glycan_score', 'glycan_coverage'])


class ScoreSet(_ScoreSet):
    __slots__ = ()
    packer = struct.Struct("!ffff")

    def __len__(self):
        return 4

    def __lt__(self, other):
        if self.glycopeptide_score < other.glycopeptide_score:
            return True
        elif abs(self.glycopeptide_score - other.glycopeptide_score) > 1e-3:
            return False

        if self.peptide_score < other.peptide_score:
            return True
        elif abs(self.peptide_score - other.peptide_score) > 1e-3:
            return False

        if self.glycan_score < other.glycan_score:
            return True
        elif abs(self.glycan_score - other.glycan_score) > 1e-3:
            return False

        if self.glycan_coverage < other.glycan_coverage:
            return True
        return False

    def __gt__(self, other):
        if self.glycopeptide_score > other.glycopeptide_score:
            return True
        elif abs(self.glycopeptide_score - other.glycopeptide_score) > 1e-3:
            return False

        if self.peptide_score > other.peptide_score:
            return True
        elif abs(self.peptide_score - other.peptide_score) > 1e-3:
            return False

        if self.glycan_score > other.glycan_score:
            return True
        elif abs(self.glycan_score - other.glycan_score) > 1e-3:
            return False

        if self.glycan_coverage > other.glycan_coverage:
            return True
        return False

    @classmethod
    def from_spectrum_matcher(cls, match):
        return cls(match.score, match.peptide_score(), match.glycan_score(), match.glycan_coverage())

    def pack(self):
        return self.packer.pack(*self)

    @classmethod
    def unpack(cls, binary):
        return cls(*cls.packer.unpack(binary))


class FDRSet(make_struct("FDRSet", ['total_q_value', 'peptide_q_value', 'glycan_q_value', 'glycopeptide_q_value'])):
    __slots__ = ()
    packer = struct.Struct("!ffff")

    def pack(self):
        return self.packer.pack(*self)

    @classmethod
    def unpack(cls, binary):
        return cls(*cls.packer.unpack(binary))

    @classmethod
    def default(cls):
        return cls(1.0, 1.0, 1.0, 1.0)

    def __lt__(self, other):
        if self.total_q_value < other.total_q_value:
            return True
        elif abs(self.total_q_value - other.total_q_value) > 1e-3:
            return False

        if self.peptide_q_value < other.peptide_q_value:
            return True
        elif abs(self.peptide_q_value - other.peptide_q_value) > 1e-3:
            return False

        if self.glycan_q_value < other.glycan_q_value:
            return True
        elif abs(self.glycan_q_value - other.glycan_q_value) > 1e-3:
            return False

        if self.glycopeptide_q_value < other.glycopeptide_q_value:
            return True
        return False

    def __gt__(self, other):
        if self.total_q_value > other.total_q_value:
            return True
        elif abs(self.total_q_value - other.total_q_value) > 1e-3:
            return False

        if self.peptide_q_value > other.peptide_q_value:
            return True
        elif abs(self.peptide_q_value - other.peptide_q_value) > 1e-3:
            return False

        if self.glycan_q_value > other.glycan_q_value:
            return True
        elif abs(self.glycan_q_value - other.glycan_q_value) > 1e-3:
            return False

        if self.glycopeptide_q_value > other.glycopeptide_q_value:
            return True
        return False


class MultiScoreSpectrumMatch(SpectrumMatch):
    __slots__ = ('score_set', 'match_type', '_q_value_set')

    def __init__(self, scan, target, score_set, best_match=False, data_bundle=None,
                 q_value_set=None, id=None, mass_shift=None, match_type=None):
        if q_value_set is None:
            q_value_set = FDRSet.default()
        else:
            q_value_set = FDRSet(*q_value_set)
        self._q_value_set = None
        super(MultiScoreSpectrumMatch, self).__init__(
            scan, target, score_set[0], best_match, data_bundle, q_value_set[0],
            id, mass_shift)
        self.score_set = ScoreSet(*score_set)
        self.q_value_set = q_value_set
        self.match_type = SpectrumMatchClassification[match_type]

    def is_multiscore(self):
        return True

    @property
    def q_value_set(self):
        return self._q_value_set

    @q_value_set.setter
    def q_value_set(self, value):
        self._q_value_set = value
        self.q_value = self._q_value_set.total_q_value

    def clone(self):
        """Create a shallow copy of this object

        Returns
        -------
        :class:`SpectrumMatch`
        """
        return self.__class__(
            self.scan, self.target, self.score_set, self.best_match, self.data_bundle,
            self.q_value_set, self.id, self.mass_shift, self.match_type)

    def __reduce__(self):
        return self.__class__, (self.scan, self.target, self.score_set, self.best_match,
                                self.data_bundle, self.q_value_set, self.id, self.mass_shift,
                                self.match_type.value)

    def pack(self):
        return (self.target.id, self.score_set.pack(), int(self.best_match),
                self.mass_shift.name, self.match_type.value)

    @classmethod
    def from_match_solution(cls, match):
        try:
            return cls(match.scan, match.target, ScoreSet.from_spectrum_matcher(match), mass_shift=match.mass_shift)
        except AttributeError:
            if isinstance(match, MultiScoreSpectrumMatch):
                return match
            else:
                raise

try:
    from glycan_profiling._c.tandem.spectrum_match import ScoreSet, FDRSet
    _has_c = True
except ImportError:
    _has_c = False

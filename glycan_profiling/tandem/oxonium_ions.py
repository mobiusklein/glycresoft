from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, Composition


class _mass_wrapper(object):

    def __init__(self, mass, annotation=None):
        self.value = mass
        self.annotation = annotation if annotation is not None else mass

    def mass(self, *args, **kwargs):
        return self.value

    def __repr__(self):
        return "MassWrapper(%f, %s)" % (self.value, self.annotation)


_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_neuac = FrozenMonosaccharideResidue.from_iupac_lite('NeuAc')

_standard_oxonium_ions = [
    _hexnac,
    _hexose,
    _mass_wrapper(_hexose.mass() - Composition("H2O").mass),
    _neuac,
    _mass_wrapper(_neuac.mass() - Composition("H2O").mass),
    FrozenMonosaccharideResidue.from_iupac_lite("Fuc"),
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - Composition("H2O").mass),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]

standard_oxonium_ions = _standard_oxonium_ions[:]

_gscore_oxonium_ions = [
    _hexnac,
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - Composition("H2O").mass),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]


class OxoniumIonScanner(object):

    def __init__(self, ions_to_search=None):
        if ions_to_search is None:
            ions_to_search = _standard_oxonium_ions
        self.ions_to_search = ions_to_search

    def scan(self, peak_list, charge=0, error_tolerance=2e-5):
        matches = []
        for ion in self.ions_to_search:
            match = peak_list.has_peak(
                ion.mass(charge=charge), error_tolerance)
            if match is not None:
                matches.append(match)
        return matches

    def ratio(self, peak_list, charge=0, error_tolerance=2e-5):
        maximum = max(p.intensity for p in peak_list)
        oxonium = sum(
            p.intensity / maximum for p in self.scan(peak_list, charge, error_tolerance))
        n = len(self.ions_to_search)
        return oxonium / n

    def __call__(self, peak_list, charge=0, error_tolerance=2e-5):
        return self.ratio(peak_list, charge, error_tolerance)


oxonium_detector = OxoniumIonScanner()
gscore_scanner = OxoniumIonScanner(_gscore_oxonium_ions)

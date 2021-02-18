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
_neugc = FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")
_water = Composition("H2O").mass

_standard_oxonium_ions = [
    _hexnac,
    _hexose,
    _mass_wrapper(_hexose.mass() - _water),
    _neuac,
    _mass_wrapper(_neuac.mass() - _water),
    FrozenMonosaccharideResidue.from_iupac_lite("Fuc"),
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - _water),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]

standard_oxonium_ions = _standard_oxonium_ions[:]

_gscore_oxonium_ions = [
    _hexnac,
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - _water),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]


class OxoniumIonScanner(object):

    def __init__(self, ions_to_search=None):
        if ions_to_search is None:
            ions_to_search = _standard_oxonium_ions
        self.ions_to_search = ions_to_search

    def scan(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        matches = []
        for ion in self.ions_to_search:
            if ion.mass() < minimum_mass:
                continue
            match = peak_list.has_peak(
                ion.mass(charge=charge), error_tolerance)
            if match is not None:
                matches.append(match)
        return matches

    def ratio(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        try:
            maximum = max(p.intensity for p in peak_list)
        except ValueError:
            return 0
        n = len([i for i in self.ions_to_search if i.mass() > minimum_mass])
        if n == 0:
            return 0
        oxonium = sum(
            p.intensity / maximum for p in self.scan(
                peak_list, charge, error_tolerance, minimum_mass))
        return oxonium / n

    def __call__(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        return self.ratio(peak_list, charge, error_tolerance, minimum_mass)


oxonium_detector = OxoniumIonScanner()
gscore_scanner = OxoniumIonScanner(_gscore_oxonium_ions)


class SignatureSpecification(object):
    __slots__ = ('components', 'masses', '_hash')

    def __init__(self, components, masses):
        self.components = tuple(
            FrozenMonosaccharideResidue.from_iupac_lite(k) for k in components)
        self.masses = tuple(masses)
        self._hash = hash(self.components)

    def __getitem__(self, i):
        return self.components[i]

    def __iter__(self):
        return iter(self.components)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.components == other.components

    def __repr__(self):
        return "{self.__class__.__name__}({self.components}, {self.masses})".format(self=self)

    def is_expected(self, glycan_composition):
        is_expected = glycan_composition._getitem_fast(self[0]) != 0
        if is_expected:
            is_expected = all(glycan_composition._getitem_fast(
                k) != 0 for k in self)
        return is_expected

    def count_of(self, glycan_composition):
        limit = float('inf')
        for component in self:
            cnt = glycan_composition._getitem_fast(component)
            if cnt < limit:
                limit = cnt
        return limit


single_signatures = {
    SignatureSpecification((str(_neuac), ), [
        _neuac.mass(),
        _neuac.mass() - _water
    ]): 0.5,
    SignatureSpecification((str(_neugc), ), [
        _neugc.mass(),
        _neugc.mass() - _water
    ]): 0.5,
}


compound_signatures = {
    SignatureSpecification(('@phosphate', 'Hex'), [
        242.01915393925,
        224.00858925555,
    ]): 0.5,
    SignatureSpecification(("@acetyl", "NeuAc"), [
        333.10598119017,
        315.09541650647
    ]): 0.5
}

import os
import re
from functools import partial

import click

from sqlalchemy.exc import OperationalError
from brainpy import periodic_table
from ms_deisotope.averagine import (
    Averagine, glycan as n_glycan_averagine, permethylated_glycan,
    peptide, glycopeptide)

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    SampleRun, Analysis)
from glycan_profiling.database.builder.glycan import GlycanCompositionHypothesisMerger
from glycan_profiling.database.builder.glycopeptide.proteomics import mzid_proteome
from glycan_profiling.chromatogram_tree import (
    MassShift, CompoundMassShift, Formate, Ammonium,
    Sodiated)

from glycan_profiling.tandem.glycopeptide.scoring import (
    binomial_score, simple_score, coverage_weighted_binomial, SpectrumMatcherBase)

from glycopeptidepy.utils.collectiontools import decoratordict
from glycopeptidepy.structure.modification import ModificationTable

from glypy import Substituent, Composition

glycan_source_validators = decoratordict()


class ModificationValidator(object):
    """Determines whether a provided command line argument can be mapped to a
    valid peptide modification in a default contructed
    :class:`glycopeptidepy.structure.modification.ModificationTable`

    Attributes
    ----------
    table : glycopeptidepy.structure.modification.ModificationTable
        The modification database to look up modifications by name.
    """
    def __init__(self):
        self.table = ModificationTable()

    def validate(self, modification_string):
        try:
            return self.table[modification_string]
        except:
            return False


class GlycanSourceValidatorBase(DatabaseBoundOperation):
    def __init__(self, database_connection, source, source_type, source_identifier=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.source = source
        self.source_type = source_type
        self.source_identifier = source_identifier

    def validate(self):
        raise NotImplementedError()


@glycan_source_validators('text')
@glycan_source_validators('combinatorial')
class TextGlycanSourceValidator(GlycanSourceValidatorBase):
    def __init__(self, database_connection, source, source_type, source_identifier=None):
        super(TextGlycanSourceValidator, self).__init__(
            database_connection, source, source_type, source_identifier)

    def validate(self):
        return os.path.exists(self.source)


@glycan_source_validators("hypothesis")
class HypothesisGlycanSourceValidator(GlycanSourceValidatorBase):
    def __init__(self, database_connection, source, source_type, source_identifier=None):
        super(HypothesisGlycanSourceValidator, self).__init__(
            database_connection, source, source_type, source_identifier)
        self.handle = DatabaseBoundOperation(source)

    def validate(self):
        if self.source_identifier is None:
            click.secho("No value passed through --glycan-source-identifier.", fg='magenta')
            return False
        try:
            hypothesis_id = int(self.source_identifier)
            inst = self.handle.query(GlycanHypothesis).get(hypothesis_id)
            return inst is not None
        except TypeError:
            hypothesis_name = self.source
            inst = self.handle.query(GlycanHypothesis).filter(GlycanHypothesis.name == hypothesis_name).first()
            return inst is not None


class GlycanHypothesisCopier(GlycanCompositionHypothesisMerger):
    pass


def validate_glycan_source(context, database_connection, source, source_type, source_identifier=None):
    glycan_source_validator_type = glycan_source_validators[source_type]
    glycan_validator = glycan_source_validator_type(database_connection, source, source_type, source_identifier)
    if not glycan_validator.validate():
        click.secho("Could not validate glycan source %s of type %s" % (
            source, source_type), fg='yellow')
        raise click.Abort()


def validate_modifications(context, modifications):
    mod_validator = ModificationValidator()
    for mod in modifications:
        if not mod_validator.validate(mod):
            click.secho("Invalid modification '%s'" % mod, fg='yellow')
            raise click.Abort()


def validate_unique_name(context, database_connection, name, klass):
    handle = DatabaseBoundOperation(database_connection)
    obj = handle.query(klass).filter(
        klass.name == name).first()
    if obj is not None:
        return klass.make_unique_name(handle.session, name)
    else:
        return name


validate_glycopeptide_hypothesis_name = partial(validate_unique_name, klass=GlycopeptideHypothesis)
validate_glycan_hypothesis_name = partial(validate_unique_name, klass=GlycanHypothesis)
validate_sample_run_name = partial(validate_unique_name, klass=SampleRun)
validate_analysis_name = partial(validate_unique_name, klass=Analysis)


def validate_mzid_proteins(context, mzid_file, target_proteins=tuple(), target_proteins_re=tuple()):
    all_proteins = set(mzid_proteome.protein_names(mzid_file))
    accepted_target_proteins = set()
    for prot in target_proteins:
        if prot in all_proteins:
            accepted_target_proteins.add(prot)
        else:
            click.secho("Could not find protein '%s'" % prot, fg='yellow')
    for prot_re in target_proteins_re:
        pat = re.compile(prot_re)
        hits = 0
        for prot in all_proteins:
            if pat.search(prot):
                accepted_target_proteins.add(prot)
                hits += 1

        if hits == 0:
            click.secho("Pattern '%s' did not match any proteins" % prot_re, fg='yellow')
    if len(accepted_target_proteins) == 0:
        click.secho("Using all proteins", fg='yellow')
    return list(accepted_target_proteins)


named_reductions = {
    'reduced': 'H2',
    'deuteroreduced': 'HH[2]'
}


def validate_reduction(context, reduction_string):
    if reduction_string is None:
        return True
    try:
        if reduction_string in named_reductions:
            return True
        else:
            if len(Composition(str(reduction_string))) > 0:
                return True
    except:
        click.secho("Could not validate reduction '%s'" % reduction_string)
        raise click.Abort()


def validate_derivatization(context, derivatization_string):
    if derivatization_string is None:
        return True
    subst = Substituent(derivatization_string)
    if len(subst.composition) == 0:
        click.secho("Could not validate derivatization '%s'" % derivatization_string)
        raise click.Abort()
    return True


def validate_element(element):
    valid = element in periodic_table
    if not valid:
        raise click.Abort("%r is not a valid element" % element)
    return valid


def parse_averagine_formula(formula):
    return Averagine({k: float(v) for k, v in re.findall(r"([A-Z][a-z]*)([0-9\.]*)", formula)
                      if float(v or 0) > 0 and validate_element(k)})


averagines = {
    'glycan': n_glycan_averagine,
    'permethylated-glycan': permethylated_glycan,
    'peptide': peptide,
    'glycopeptide': glycopeptide
}


def validate_averagine(averagine_string):
    if averagine_string in averagines:
        return averagines[averagine_string]
    else:
        return parse_averagine_formula(averagine_string)


adducts = {
    "ammonium": Ammonium,
    "formate": Formate,
    "sodium": Sodiated,
}


def validate_adduct(adduct_string, multiplicity=1):
    multiplicity = int(multiplicity)
    if adduct_string.lower() in adducts:
        return (adducts[adduct_string.lower()], multiplicity)
    else:
        try:
            composition = Composition(adduct_string)
            shift = MassShift(adduct_string, composition)
            return (shift, multiplicity)
        except Exception as e:
            print(e)
            click.secho("Could not validate adduct %s" % (adduct_string,), fg='yellow')
            raise click.Abort()


glycopeptide_tandem_scoring_functions = {
    "binomial": binomial_score.BinomialSpectrumMatcher,
    "simple": simple_score.SimpleCoverageScorer,
    "coverage_weighted_binomial": coverage_weighted_binomial.CoverageWeightedBinomialScorer
}


def validate_glycopeptide_tandem_scoring_function(context, name):
    if isinstance(name, SpectrumMatcherBase):
        return name
    try:
        return glycopeptide_tandem_scoring_functions[name]
    except KeyError:
        raise click.Abort("Could not recognize scoring function by name %r" % (name,))


def get_by_name_or_id(session, model_type, name_or_id):
    try:
        object_id = int(name_or_id)
        inst = session.query(model_type).get(object_id)
        if inst is None:
            raise ValueError("No instance of type %s with id %r" %
                             (model_type, name_or_id))
        return inst
    except ValueError:
        try:
            inst = session.query(model_type).filter(
                model_type.name == name_or_id).one()
            return inst
        except:
            raise click.BadParameter("Could not locate an instance of %r with identifier %r" % (
                model_type.__name__, name_or_id))


def validate_database_unlocked(database_connection):
    try:
        db = DatabaseBoundOperation(database_connection)
        db.session.add(GlycanHypothesis(name="_____not_real_do_not_use______"))
        db.session.rollback()
        return True
    except OperationalError:
        return False

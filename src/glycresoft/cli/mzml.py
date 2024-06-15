import os
import functools

from collections import defaultdict

import click

import ms_deisotope
import glypy

from ms_deisotope import MSFileLoader
from ms_deisotope.output import ProcessedMSFileLoader
from ms_deisotope.feature_map import quick_index
from ms_deisotope.tools.deisotoper import deisotope

from glycresoft.cli.base import cli

from glycresoft.profiler import SampleConsumer


@cli.group('mzml', short_help='Inspect and preprocess mzML files')
def mzml_cli():
    pass


preprocess = click.Command(
    "preprocess",
    context_settings=deisotope.context_settings,
    callback=functools.partial(deisotope.callback, workflow_cls=SampleConsumer, configure_logging=False),
    params=deisotope.params,
    help=deisotope.help,
    epilog=deisotope.epilog,
)

mzml_cli.add_command(preprocess, "preprocess")


@mzml_cli.command('rt-to-id', short_help="Look up the retention time for a given scan id")
@click.argument("ms-file", type=click.Path(exists=True))
@click.argument("rt", type=float)
def rt_to_id(ms_file, rt):
    loader = MSFileLoader(ms_file)
    id = loader._locate_ms1_scan(loader.get_scan_by_time(rt)).id
    click.echo(id)


@mzml_cli.command("info", short_help='Summary information describing a processed mzML file')
@click.argument("ms-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def msfile_info(ms_file):
    reader = ProcessedMSFileLoader(ms_file)
    if not reader.has_index_file():
        index, intervals = quick_index.index(ms_deisotope.MSFileLoader(ms_file))
        reader.extended_index = index
        with open(reader._index_file_name, 'w') as handle:
            index.serialize(handle)
    click.echo("Name: %s" % (os.path.basename(ms_file),))
    click.echo("MS1 Scans: %d" % (len(reader.extended_index.ms1_ids),))
    click.echo("MSn Scans: %d" % (len(reader.extended_index.msn_ids),))

    n_defaulted = 0
    n_orphan = 0

    charges = defaultdict(int)
    first_msn = float('inf')
    last_msn = 0
    for scan_info in reader.extended_index.msn_ids.values():
        n_defaulted += scan_info.get('defaulted', False)
        n_orphan += scan_info.get('orphan', False)
        charges[scan_info['charge']] += 1
        rt = scan_info['scan_time']
        if rt < first_msn:
            first_msn = rt
        if rt > last_msn:
            last_msn = rt

    click.echo("First MSn Scan: %0.2f Minutes" % (first_msn,))
    click.echo("Last MSn Scan: %0.2f Minutes" % (last_msn,))

    for charge, count in sorted(charges.items()):
        if not isinstance(charge, int):
            continue
        click.echo("Precursors with Charge State %d: %d" % (charge, count))

    click.echo("Defaulted MSn Scans: %d" % (n_defaulted,))
    click.echo("Orphan MSn Scans: %d" % (n_orphan,))


@mzml_cli.command("oxonium-signature", short_help=(
    'Report Oxonium Ion Signature Score (G-Score) and Glycan Structure Signature Score'
    ' for each MSn scan in a processed mzML file'))
@click.argument("ms-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-g", "--g-score-threshold", type=float, default=0.05, help="Minimum G-Score to report")
def oxonium_signature(ms_file, g_score_threshold=0.05):
    reader = ProcessedMSFileLoader(ms_file)
    if not reader.has_index_file():
        click.secho("Building temporary index...", fg='yellow')
        index, intervals = quick_index.index(ms_deisotope.MSFileLoader(ms_file))
        reader.extended_index = index
        with open(reader._index_file_name, 'w') as handle:
            index.serialize(handle)

    from glycresoft.tandem.glycan.scoring.signature_ion_scoring import SignatureIonScorer
    from glycresoft.tandem.oxonium_ions import gscore_scanner

    refcomp = glypy.GlycanComposition.parse("{Fuc:1; Hex:5; HexNAc:4; Neu5Ac:1; NeuGc:1}")
    headers = ["scan_id", "precursor_mass", "precursor_charge", "g_score", "signature_score"]
    click.echo("\t".join(headers))
    for scan_id in reader.extended_index.msn_ids.keys():
        scan = reader.get_scan_by_id(scan_id)
        gscore = gscore_scanner(scan.deconvoluted_peak_set)
        if gscore >= g_score_threshold:
            signature_match = SignatureIonScorer.evaluate(
                scan, refcomp, use_oxonium=False)
            click.echo("%s\t%f\t%r\t%f\t%f" % (
                scan_id, scan.precursor_information.neutral_mass,
                scan.precursor_information.charge, gscore,
                signature_match.score))

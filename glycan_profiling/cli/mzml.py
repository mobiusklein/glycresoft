from collections import defaultdict

import click
import os

import ms_peak_picker
import ms_deisotope
import glypy

from ms_deisotope import MSFileLoader
from ms_deisotope.data_source import RandomAccessScanSource
from ms_deisotope.output.mzml import ProcessedMzMLDeserializer
from ms_deisotope.feature_map import quick_index

from glycan_profiling.cli.base import cli, HiddenOption, processes_option
from glycan_profiling.cli.validators import (
    AveragineParamType)

from glycan_profiling.profiler import (
    SampleConsumer,
    ThreadedMzMLScanCacheHandler)


@cli.group('mzml', short_help='Inspect and preprocess mzML files')
def mzml_cli():
    pass


@mzml_cli.command('rt-to-id', short_help="Look up the retention time for a given scan id")
@click.argument("ms-file", type=click.Path(exists=True))
@click.argument("rt", type=float)
def rt_to_id(ms_file, rt):
    loader = MSFileLoader(ms_file)
    id = loader._locate_ms1_scan(loader.get_scan_by_time(rt)).id
    click.echo(id)


@mzml_cli.command("preprocess", short_help=(
    "Convert raw mass spectra data into deisotoped neutral mass peak lists written to mzML."
    " Can accept mzML or mzXML with either profile or centroided scans."))
@click.argument("ms-file", type=click.Path(exists=True), doc_help=(
    "Path to an mass spectral data file in one of the supported formats"))
@click.argument("outfile-path", type=click.Path(writable=True), doc_help=(
    "Path to write the processed output to"))
@click.option("-a", "--averagine", default=["glycan"],
              type=AveragineParamType(),
              help='Averagine model to use for MS1 scans. Either a name or formula',
              multiple=True)
@click.option("-an", "--msn-averagine", default="peptide",
              type=AveragineParamType(),
              help='Averagine model to use for MS^n scans. Either a name or formula')
@click.option("-s", "--start-time", type=float, default=0.0,
              help='Scan time to begin processing at in minutes')
@click.option("-e", "--end-time", type=float, default=float('inf'),
              help='Scan time to stop processing at in minutes')
@click.option("-c", "--maximum-charge", type=int, default=8,
              help=('Highest absolute charge state to consider'))
@click.option("-n", "--name", default=None,
              help="Name for the sample run to be stored. Defaults to the base name of the input mzML file")
@click.option("-t", "--score-threshold", type=float, default=SampleConsumer.MS1_SCORE_THRESHOLD,
              help="Minimum score to accept an isotopic pattern fit in an MS1 scan")
@click.option("-tn", "--msn-score-threshold", type=float, default=SampleConsumer.MSN_SCORE_THRESHOLD,
              help="Minimum score to accept an isotopic pattern fit in an MS^n scan")
@click.option("-m", "--missed-peaks", type=int, default=3,
              help="Number of missing peaks to permit before an isotopic fit is discarded")
@click.option("-mn", "--msn-missed-peaks", type=int, default=1,
              help="Number of missing peaks to permit before an isotopic fit is discarded in an MSn scan")
@processes_option
@click.option("-b", "--background-reduction", type=float, default=5., help=(
              "Background reduction factor. Larger values more aggresively remove low abundance"
              " signal in MS1 scans."))
@click.option("-bn", "--msn-background-reduction", type=float, default=0., help=(
              "Background reduction factor. Larger values more aggresively remove low abundance"
              " signal in MS^n scans."))
@click.option("-r", '--transform', multiple=True, type=click.Choice(
    sorted(ms_peak_picker.scan_filter.filter_register.keys())),
    help="Scan transformations to apply to MS1 scans. May specify more than once.")
@click.option("-rn", '--msn-transform', multiple=True, type=click.Choice(
    sorted(ms_peak_picker.scan_filter.filter_register.keys())),
    help="Scan transformations to apply to MS^n scans. May specify more than once.")
@click.option("-v", "--extract-only-tandem-envelopes", is_flag=True, default=False,
              help='Only work on regions that will be chosen for MS/MS')
@click.option("-g", "--ms1-averaging", default=0, type=int, help=(
    "The number of MS1 scans before and after the current MS1 "
    "scan to average when picking peaks."))
@click.option("--ignore-msn", is_flag=True, default=False, help="Ignore MS^n scans")
@click.option("--profile", default=False, is_flag=True, help=(
              "Force profile scan configuration."), cls=HiddenOption)
@click.option("-i", "--isotopic-strictness", default=2.0, type=float, cls=HiddenOption)
@click.option("-in", "--msn-isotopic-strictness", default=0.0, type=float, cls=HiddenOption)
@click.option("-snr", "--signal-to-noise-threshold", default=1.0, type=float, help=(
    "Signal-to-noise ratio threshold to apply when filtering peaks"))
@click.option("-mo", "--mass-offset", default=0.0, type=float, help=("Shift peak masses by the given amount"))
def preprocess(ms_file, outfile_path, averagine=None, start_time=None, end_time=None, maximum_charge=None,
               name=None, msn_averagine=None, score_threshold=35., msn_score_threshold=10., missed_peaks=1,
               msn_missed_peaks=1, background_reduction=5., msn_background_reduction=0.,
               transform=None, msn_transform=None, processes=4, extract_only_tandem_envelopes=False,
               ignore_msn=False, profile=False, isotopic_strictness=2.0, ms1_averaging=0,
               msn_isotopic_strictness=0.0, signal_to_noise_threshold=1.0, mass_offset=0.0, deconvolute=True):
    '''Convert raw mass spectra data into deisotoped neutral mass peak lists written to mzML.
    '''
    if transform is None:
        transform = []
    if msn_transform is None:
        msn_transform = []

    if (ignore_msn and extract_only_tandem_envelopes):
        click.secho(
            "Cannot use both --ignore-msn and --extract-only-tandem-envelopes",
            fg='red')
        raise click.Abort("Cannot use both --ignore-msn and --extract-only-tandem-envelopes")

    cache_handler_type = ThreadedMzMLScanCacheHandler
    click.echo("Preprocessing %s" % ms_file)
    minimum_charge = 1 if maximum_charge > 0 else -1
    charge_range = (minimum_charge, maximum_charge)

    loader = MSFileLoader(ms_file)
    if isinstance(loader, RandomAccessScanSource):
        last_scan = loader[len(loader) - 1]
        last_time = last_scan.scan_time

        if end_time > last_time:
            end_time = last_time

        try:
            start_scan = loader._locate_ms1_scan(loader.get_scan_by_time(start_time))
        except IndexError:
            start_scan = loader.get_scan_by_time(start_time)

        try:
            end_scan = loader._locate_ms1_scan(loader.get_scan_by_time(end_time))
        except IndexError:
            end_scan = loader.get_scan_by_time(end_time)

        start_scan_id = start_scan.id
        end_scan_id = end_scan.id

        start_scan_time = start_scan.scan_time
        end_scan_time = end_scan.scan_time

        loader.reset()
        loader.start_from_scan(
            start_scan_id, require_ms1=loader.has_ms1_scans(), grouped=True)
    else:
        click.secho("The file format provided does not support random"
                    " access, start and end points will be ignored", fg='yellow')
        start_scan_time = 0
        start_scan_id = None

        end_scan_time = float('inf')
        end_scan_id = None
        loader.make_iterator(grouped=True)

    first_bunch = next(loader)
    if first_bunch.precursor is not None:
        is_profile = (first_bunch.precursor.is_profile or profile)
    elif first_bunch.products:
        is_profile = (first_bunch.products[0].is_profile or profile)

    if is_profile:
        click.secho("Spectra are profile")
    else:
        click.secho("Spectra are centroided", fg='yellow')

    if name is None:
        name = os.path.splitext(os.path.basename(ms_file))[0]

    if os.path.exists(outfile_path) and not os.access(outfile_path, os.W_OK):
        click.secho("Can't write to output file path", fg='red')
        raise click.Abort()

    click.secho("Initializing %s" % name, fg='green')
    click.echo("from %s (%0.2f) to %s (%0.2f)" % (
        start_scan_id, start_scan_time, end_scan_id, end_scan_time))
    if deconvolute:
        click.echo("charge range: %s" % (charge_range,))

    if is_profile:
        ms1_peak_picking_args = {
            "transforms": [
            ] + list(transform),
            "signal_to_noise_threshold": signal_to_noise_threshold
        }
        if background_reduction:
            ms1_peak_picking_args['transforms'].append(
                ms_peak_picker.scan_filter.FTICRBaselineRemoval(
                    scale=background_reduction, window_length=2))
            ms1_peak_picking_args['transforms'].append(ms_peak_picker.scan_filter.SavitskyGolayFilter())
    else:
        ms1_peak_picking_args = {
            "transforms": [
                ms_peak_picker.scan_filter.FTICRBaselineRemoval(
                    scale=background_reduction, window_length=2),
            ] + list(transform)
        }

    if msn_background_reduction > 0.0:
        msn_peak_picking_args = {
            "transforms": [
                ms_peak_picker.scan_filter.FTICRBaselineRemoval(
                    scale=msn_background_reduction, window_length=2),
            ] + list(msn_transform)
        }
    else:
        msn_peak_picking_args = {
            "transforms": [
            ] + list(msn_transform)
        }

    if mass_offset != 0.0:
        ms1_peak_picking_args['transforms'].append(
            ms_peak_picker.scan_filter.RecalibrateMass(offset=mass_offset))
        msn_peak_picking_args['transforms'].append(
            ms_peak_picker.scan_filter.RecalibrateMass(offset=mass_offset))

    if deconvolute:
        if len(averagine) == 1:
            averagine = averagine[0]
            ms1_deconvoluter_type = ms_deisotope.deconvolution.AveraginePeakDependenceGraphDeconvoluter
        else:
            ms1_deconvoluter_type = ms_deisotope.deconvolution.MultiAveraginePeakDependenceGraphDeconvoluter

        ms1_deconvolution_args = {
            "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(score_threshold, isotopic_strictness),
            "max_missed_peaks": missed_peaks,
            "averagine": averagine,
            "truncate_after": SampleConsumer.MS1_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MS1_IGNORE_BELOW,
            "deconvoluter_type": ms1_deconvoluter_type
        }

        if msn_isotopic_strictness >= 1:
            msn_isotopic_scorer = ms_deisotope.scoring.PenalizedMSDeconVFitter(
                msn_score_threshold, msn_isotopic_strictness)
        else:
            msn_isotopic_scorer = ms_deisotope.scoring.MSDeconVFitter(msn_score_threshold)

        msn_deconvolution_args = {
            "scorer": msn_isotopic_scorer,
            "averagine": msn_averagine,
            "max_missed_peaks": msn_missed_peaks,
            "truncate_after": SampleConsumer.MSN_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MSN_IGNORE_BELOW
        }
    else:
        ms1_deconvolution_args = None
        msn_deconvolution_args = None

    consumer = SampleConsumer(
        ms_file,
        ms1_peak_picking_args=ms1_peak_picking_args,
        ms1_deconvolution_args=ms1_deconvolution_args,
        msn_peak_picking_args=msn_peak_picking_args,
        msn_deconvolution_args=msn_deconvolution_args,
        storage_path=outfile_path, sample_name=name,
        start_scan_id=start_scan_id, cache_handler_type=cache_handler_type,
        end_scan_id=end_scan_id, n_processes=processes,
        extract_only_tandem_envelopes=extract_only_tandem_envelopes,
        ignore_tandem_scans=ignore_msn,
        ms1_averaging=ms1_averaging,
        deconvolute=deconvolute)
    consumer.display_header()
    consumer.start()


@mzml_cli.command("info", short_help='Summary information describing a processed mzML file')
@click.argument("ms-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def msfile_info(ms_file):
    reader = ProcessedMzMLDeserializer(ms_file)
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
    reader = ProcessedMzMLDeserializer(ms_file)
    if not reader.has_index_file():
        click.secho("Building temporary index...", fg='yellow')
        index, intervals = quick_index.index(ms_deisotope.MSFileLoader(ms_file))
        reader.extended_index = index
        with open(reader._index_file_name, 'w') as handle:
            index.serialize(handle)

    from glycan_profiling.tandem.glycan.scoring.signature_ion_scoring import SignatureIonScorer
    from glycan_profiling.tandem.oxonium_ions import gscore_scanner
    refcomp = glypy.GlycanComposition.parse("{Fuc:1; Hex:5; HexNAc:4; Neu5Ac:2}")
    for scan_id in reader.extended_index.msn_ids.keys():
        scan = reader.get_scan_by_id(scan_id)
        gscore = gscore_scanner(scan.deconvoluted_peak_set)
        if gscore >= g_score_threshold:
            signature_match = SignatureIonScorer.evaluate(scan, refcomp)
            click.echo("%s\t%f\t%r\t%f\t%f" % (
                scan_id, scan.precursor_information.neutral_mass,
                scan.precursor_information.charge, gscore,
                signature_match.score))


@mzml_cli.command("peak-picking", short_help=(
    "Convert raw mass spectra data into centroid peak lists written to mzML."
    " Can accept mzML or mzXML with either profile or centroided scans."))
@click.argument("ms-file", type=click.Path(exists=True))
@click.argument("outfile-path", type=click.Path(writable=True))
@processes_option
@click.option("-b", "--background-reduction", type=float, default=5., help=(
              "Background reduction factor. Larger values more aggresively remove low abundance"
              " signal in MS1 scans."))
@click.option("-bn", "--msn-background-reduction", type=float, default=0., help=(
              "Background reduction factor. Larger values more aggresively remove low abundance"
              " signal in MS^n scans."))
@click.option("-r", '--transform', multiple=True, type=click.Choice(
    sorted(ms_peak_picker.scan_filter.filter_register.keys())),
    help="Scan transformations to apply to MS1 scans. May specify more than once.")
@click.option("-rn", '--msn-transform', multiple=True, type=click.Choice(
    sorted(ms_peak_picker.scan_filter.filter_register.keys())),
    help="Scan transformations to apply to MS^n scans. May specify more than once.")
@click.option("-v", "--extract-only-tandem-envelopes", is_flag=True, default=False,
              help='Only work on regions that will be chosen for MS/MS')
@click.option("--profile", default=False, is_flag=True, help=(
              "Force profile scan configuration."), cls=HiddenOption)
@click.option("-s", "--start-time", type=float, default=0.0, help='Scan time to begin processing at')
@click.option("-e", "--end-time", type=float, default=float('inf'), help='Scan time to stop processing at')
@click.option("-n", "--name", default=None,
              help="Name for the sample run to be stored. Defaults to the base name of the input mzML file")
@click.option("-g", "--ms1-averaging", default=0, type=int, help=(
    "The number of MS1 scans before and after the current MS1 "
    "scan to average when picking peaks."))
@click.pass_context
def peak_picker(ctx, ms_file, outfile_path, start_time=None, end_time=None,
                name=None, background_reduction=5., msn_background_reduction=0.,
                transform=None, msn_transform=None, processes=4, extract_only_tandem_envelopes=False,
                mzml=True, profile=False, ms1_averaging=0):
    ctx.forward(preprocess, deconvolute=False)

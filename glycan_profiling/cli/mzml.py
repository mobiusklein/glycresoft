import click
import os
import multiprocessing

from glycan_profiling.cli.base import cli
from glycan_profiling.cli.validators import (
    validate_averagine, validate_sample_run_name)

import ms_deisotope
import ms_peak_picker

from ms_deisotope.processor import MSFileLoader
from glycan_profiling.chromatogram_tree import find_truncation_points
from glycan_profiling.profiler import (
    SampleConsumer,
    ThreadedDatabaseScanCacheHandler,
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
@click.argument("ms-file", type=click.Path(exists=True))
@click.argument("outfile-path", type=click.Path(writable=True))
@click.option("-a", "--averagine", default='glycan',
              help='Averagine model to use for MS1 scans. Either a name or formula.')
@click.option("-an", "--msn-averagine", default='peptide',
              help='Averagine model to use for MS^n scans. Either a name or formula.')
@click.option("-s", "--start-time", type=float, default=0.0, help='Scan time to begin processing at')
@click.option("-e", "--end-time", type=float, default=float('inf'), help='Scan time to stop processing at')
@click.option("-c", "--maximum-charge", type=int, default=8,
              help=('Highest charge state considered. '
                    'Defaults to 8'))
@click.option("-n", "--name", default=None,
              help="Name for the sample run to be stored. Defaults to the base name of the input mzML file")
@click.option("-t", "--score-threshold", type=float, default=35.,
              help="Minimum score to accept an isotopic pattern fit in an MS1 scan. Scales with intensity.")
@click.option("-tn", "--msn-score-threshold", type=float, default=10.,
              help="Minimum score to accept an isotopic pattern fit in an MS^n scan. Scales with intensity.")
@click.option("-m", "--missed-peaks", type=int, default=1,
              help="Number of missing peaks to permit before an isotopic fit is discarded")
@click.option("-p", "--processes", 'processes', type=click.IntRange(1, multiprocessing.cpu_count()),
              default=min(multiprocessing.cpu_count(), 4), help=('Number of worker processes to use. Defaults to 4 '
                                                                 'or the number of CPUs, whichever is lower'))
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
    "Force profile scan configuration."))
def preprocess(ms_file, outfile_path, averagine=None, start_time=None, end_time=None, maximum_charge=None,
               name=None, msn_averagine=None, score_threshold=35., msn_score_threshold=10., missed_peaks=1,
               background_reduction=5., msn_background_reduction=0., transform=None, msn_transform=None,
               processes=4, extract_only_tandem_envelopes=False, mzml=True, profile=False):
    if transform is None:
        transform = []
    if msn_transform is None:
        msn_transform = []
    cache_handler_type = ThreadedMzMLScanCacheHandler
    click.echo("Preprocessing %s" % ms_file)
    minimum_charge = 1 if maximum_charge > 0 else -1
    charge_range = (minimum_charge, maximum_charge)

    loader = MSFileLoader(ms_file)

    start_scan_id = loader._locate_ms1_scan(
        loader.get_scan_by_time(start_time)).id
    end_scan_id = loader._locate_ms1_scan(
        loader.get_scan_by_time(end_time)).id

    loader.reset()
    is_profile = (next(loader).precursor.is_profile or profile)
    if is_profile:
        click.secho("Spectra are profile")
    else:
        click.secho("Spectra are centroided")

    if name is None:
        name = os.path.splitext(os.path.basename(ms_file))[0]

    if os.path.exists(outfile_path) and not os.access(outfile_path, os.W_OK):
        click.secho("Can't write to output file path", fg='red')
        raise click.Abort()

    click.secho("Initializing %s" % name, fg='green')
    click.echo("from %s to %s" % (start_scan_id, end_scan_id))
    click.echo("charge range: %s" % (charge_range,))

    averagine = validate_averagine(averagine)
    msn_averagine = validate_averagine(msn_averagine)

    if is_profile:
        ms1_peak_picking_args = {
            "transforms": [
                ms_peak_picker.scan_filter.FTICRBaselineRemoval(
                    scale=background_reduction, window_length=2),
                ms_peak_picker.scan_filter.SavitskyGolayFilter()
            ] + list(transform)
        }
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

    ms1_deconvolution_args = {
        "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(score_threshold, 2.),
        "max_missed_peaks": missed_peaks,
        "averagine": averagine
    }

    msn_deconvolution_args = {
        "scorer": ms_deisotope.scoring.MSDeconVFitter(msn_score_threshold),
        "averagine": msn_averagine,
        "max_missed_peaks": missed_peaks,
    }

    consumer = SampleConsumer(
        ms_file, averagine=averagine, charge_range=charge_range,
        ms1_peak_picking_args=ms1_peak_picking_args,
        ms1_deconvolution_args=ms1_deconvolution_args,
        msn_peak_picking_args=msn_peak_picking_args,
        msn_deconvolution_args=msn_deconvolution_args,
        storage_path=outfile_path, sample_name=name,
        start_scan_id=start_scan_id, cache_handler_type=cache_handler_type,
        end_scan_id=end_scan_id, n_processes=processes,
        extract_only_tandem_envelopes=extract_only_tandem_envelopes)

    consumer.start()

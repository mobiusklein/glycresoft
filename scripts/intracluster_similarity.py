import os
import csv

from typing import Optional, DefaultDict, List, NamedTuple

import click

from glycan_profiling import serialize
from ms_deisotope.clustering.scan_clustering import SpectrumCluster
from ms_deisotope.output import ProcessedMSFileLoader
from ms_deisotope.data_source import ProcessedScan


class ClusterSpec(NamedTuple):
    charge: int
    adduct: str


class ClusterResult(NamedTuple):
    glycopeptide: str
    charge: int
    adduct: str
    average_similarity: float
    cluster_size: int


@click.command()
@click.argument("analysis_path")
@click.option("-m", "--mzml-path", default=None, type=str)
@click.option("-q", "--q-value-threshold", default=0.01, type=float)
@click.option("-o", "--output-path", default=None, type=str)
def main(analysis_path: str, mzml_path: Optional[str] = None, q_value_threshold: Optional[float] = 0.01, output_path: Optional[str]=None):
    ads = serialize.AnalysisDeserializer(analysis_path)
    if mzml_path is None:
        ms_reader = ads.open_ms_file()
    else:
        ms_reader = ProcessedMSFileLoader(mzml_path)
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(
                os.path.realpath(analysis_path)
            ),
            "cluster_similarities.csv"
        )

    idgps: List[serialize.IdentifiedGlycopeptide] = ads.query(serialize.IdentifiedGlycopeptide).filter(serialize.IdentifiedGlycopeptide.q_value <= q_value_threshold).all()
    click.echo(f"Loaded {len(idgps)} glycopeptides")
    with open(output_path, 'wt', newline='') as fh:
        writer = csv.DictWriter(fh, ClusterResult._fields)
        writer.writeheader()

        with click.progressbar(idgps) as bar:
            for idgp in bar:
                tracks: DefaultDict[ClusterSpec, List[ProcessedScan]] = DefaultDict(list)
                gp = idgp.structure
                sset: serialize.GlycopeptideSpectrumSolutionSet
                for sset in idgp.tandem_solutions:
                    scan_id = sset.scan.scan_id
                    try:
                        gpsm = sset.solution_for(gp)
                        if gpsm.q_value > q_value_threshold:
                            continue
                        adduct = gpsm.mass_shift.name
                        scan = ms_reader.get_scan_by_id(scan_id)
                        z = scan.precursor_information.charge
                        tracks[ClusterSpec(z, adduct)].append(scan)
                    except KeyError:
                        continue
                gp_str = gp.glycopeptide_sequence
                for key, track in tracks.items():
                    cluster = SpectrumCluster(track)
                    sim = cluster.average_similarity()
                    rec = ClusterResult(gp_str, *key, sim, len(track))
                    writer.writerow(rec._asdict())



if __name__ == "__main__":
    main.main()
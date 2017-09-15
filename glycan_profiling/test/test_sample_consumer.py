import unittest
import tempfile
import os
import glob

import ms_peak_picker
import ms_deisotope
from ms_deisotope.output.mzml import ProcessedMzMLDeserializer


from glycan_profiling.test import fixtures
from glycan_profiling.profiler import SampleConsumer


agp_glycomics_mzml = fixtures.get_test_data("AGP_Glycomics_20150930_06.centroid.mzML")


class SampleConsumerTest(unittest.TestCase):
    def build_args(self):
        ms1_peak_picking_args = {
            "transforms": [
            ]
        }

        msn_peak_picking_args = {
            "transforms": [
            ]
        }

        ms1_deconvolution_args = {
            "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(35.0, 2.0),
            "max_missed_peaks": 1,
            "averagine": ms_deisotope.glycan,
            "truncate_after": SampleConsumer.MS1_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MS1_IGNORE_BELOW
        }

        msn_deconvolution_args = {
            "scorer": ms_deisotope.scoring.MSDeconVFitter(10.0),
            "averagine": ms_deisotope.glycan,
            "max_missed_peaks": 1,
            "truncate_after": SampleConsumer.MSN_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MSN_IGNORE_BELOW
        }
        return (
            ms1_peak_picking_args, msn_peak_picking_args,
            ms1_deconvolution_args, msn_deconvolution_args)

    def make_output_directory(self):
        path = tempfile.mkdtemp()
        return path

    def test_consumer(self):
        (ms1_peak_picking_args, msn_peak_picking_args,
         ms1_deconvolution_args, msn_deconvolution_args) = self.build_args()
        outdir = self.make_output_directory()
        outpath = os.path.join(outdir, "test-output.mzML")

        consumer = SampleConsumer(
            agp_glycomics_mzml,
            ms1_peak_picking_args=ms1_peak_picking_args,
            ms1_deconvolution_args=ms1_deconvolution_args,
            msn_peak_picking_args=msn_peak_picking_args,
            msn_deconvolution_args=msn_deconvolution_args,
            storage_path=outpath, sample_name=None,
            n_processes=5,
            extract_only_tandem_envelopes=False)
        consumer.start()

        reader = ProcessedMzMLDeserializer(outpath)

        scan = reader.get_scan_by_id("scanId=1601016")
        self.assertIsNotNone(scan.deconvoluted_peak_set.has_peak(958.66, use_mz=1))

        reader.close()

        self.cleanup(outdir)

    def cleanup(self, directory):
        files = glob.glob(os.path.join(directory, "*"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.remove(directory)
        except OSError:
            pass


if __name__ == '__main__':
    unittest.main()

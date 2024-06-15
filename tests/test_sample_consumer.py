import unittest
import tempfile
import os
import glob

import ms_peak_picker
import ms_deisotope
from ms_deisotope.output import ProcessedMSFileLoader

import numpy as np

from glycresoft.profiler import SampleConsumer

from . import fixtures

agp_glycomics_mzml = fixtures.get_test_data("AGP_Glycomics_20150930_06.centroid.mzML")
agp_glycproteomics_mzml = fixtures.get_test_data("20150710_3um_AGP_001_29_30.mzML")
agp_glycproteomics_mzml_reference = fixtures.get_test_data("20150710_3um_AGP_001_29_30.preprocessed.mzML")


class SampleConsumerBase(object):
    def make_output_directory(self):
        path = tempfile.mkdtemp()
        return path

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


class MSMSSampleConsumerTest(unittest.TestCase, SampleConsumerBase):
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
            "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(20.0, 2.0),
            "max_missed_peaks": 3,
            "averagine": ms_deisotope.glycopeptide,
            "truncate_after": SampleConsumer.MS1_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MS1_IGNORE_BELOW
        }

        msn_deconvolution_args = {
            "scorer": ms_deisotope.scoring.MSDeconVFitter(10.0),
            "averagine": ms_deisotope.peptide,
            "max_missed_peaks": 1,
            "truncate_after": SampleConsumer.MSN_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MSN_IGNORE_BELOW
        }
        return (
            ms1_peak_picking_args, msn_peak_picking_args,
            ms1_deconvolution_args, msn_deconvolution_args)

    def test_consumer(self):
        (ms1_peak_picking_args, msn_peak_picking_args,
         ms1_deconvolution_args, msn_deconvolution_args) = self.build_args()
        outdir = self.make_output_directory()
        outpath = os.path.join(outdir, "test-output.mzML")

        consumer = SampleConsumer(
            agp_glycproteomics_mzml,
            ms1_peak_picking_args=ms1_peak_picking_args,
            ms1_deconvolution_args=ms1_deconvolution_args,
            msn_peak_picking_args=msn_peak_picking_args,
            msn_deconvolution_args=msn_deconvolution_args,
            storage_path=outpath, sample_name=None,
            n_processes=5,
            extract_only_tandem_envelopes=True,
            ms1_averaging=1)
        consumer.start()

        reader = ProcessedMSFileLoader(outpath)
        reference = ProcessedMSFileLoader(agp_glycproteomics_mzml_reference)

        for a_bunch, b_bunch in zip(reader, reference):
            assert a_bunch.precursor.id == b_bunch.precursor.id
            assert len(a_bunch.products) == len(b_bunch.products)
            for a_product, b_product in zip(a_bunch.products, b_bunch.products):
                assert a_product.precursor_information.defaulted == b_product.precursor_information.defaulted
                matched = np.isclose(a_product.precursor_information.neutral_mass,
                                     b_product.precursor_information.neutral_mass)
                message = ["%0.3f not close to %0.3f for %s of %s" % (
                    a_product.precursor_information.neutral_mass,
                    b_product.precursor_information.neutral_mass,
                    a_product.id, a_product.precursor_information.precursor_scan_id)]
                message.append("Found precursor score %r, expected %r" % (
                    a_product.precursor_information.precursor.deconvoluted_peak_set.has_peak(
                        a_product.precursor_information.neutral_mass).score,
                    b_product.precursor_information.precursor.deconvoluted_peak_set.has_peak(
                        b_product.precursor_information.neutral_mass).score
                ))
                assert matched, '\n'.join(message)
                assert len(a_product.deconvoluted_peak_set) == len(b_product.deconvoluted_peak_set)

        reader.close()
        reference.close()

        self.cleanup(outdir)


@unittest.skip("This test is subsumed by MSMSSampleConsumerTest")
class SampleConsumerTest(unittest.TestCase, SampleConsumerBase):
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

        reader = ProcessedMSFileLoader(outpath)

        scan = reader.get_scan_by_id("scanId=1601016")
        self.assertIsNotNone(scan.deconvoluted_peak_set.has_peak(958.66, use_mz=1))

        reader.close()

        self.cleanup(outdir)


if __name__ == '__main__':
    unittest.main()

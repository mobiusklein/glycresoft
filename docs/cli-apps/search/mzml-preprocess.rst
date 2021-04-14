LC-MS/MS Data Preprocessing and Deconvolution
=============================================


Convert raw mass spectral data files into deisotoped neutral mass peak lists
written to a new **mzML** [Martens2011]_ file. For tandem mass spectra,
recalculate precursor ion monoisotopic peaks.

This task is computationally intensive, and uses several collaborative processes
to share the work.

.. click:: glycan_profiling.cli.mzml:preprocess
    :prog: glycresoft mzml preprocess


Usage example
-------------

.. code-block:: bash
    :caption: example usage

    glycresoft-cli mzml preprocess -a permethylated-glycan -t 20 -p 6 \
        -s 5.0 -e 60.0 "path/to/input" "path/to/output.mzML"

Averagine Models
----------------

Argument type for ``<averagine>``. The model selected influences how isotopic
patterns are estimated for an arbitrary mass. The value of this parameter may
be a builtin model name or a formula.

For a more complete discsussion of how "averagine" isotopic models work, see [Senko1995]_.

Builtin Models
~~~~~~~~~~~~~~

.. exec::

    from glycan_profiling.cli.validators import AveragineParamType
    from rst_table import as_rest_table

    rows = [
        ("Model Name", "Formula")
    ]

    def formula(mapping):
        return ''.join(["%s%0.2g" % (k, v) for k, v in mapping.items()])

    for name, model in AveragineParamType.models.items():
        rows.append((name, formula(model)))
    
    print(as_rest_table(rows))


Supported File Formats
----------------------

``MS_FILE`` may be in mzML or mzXML format.


Signal Filters
--------------

Prior to picking peaks, the raw mass spectral signal may be filtered a number
of ways. By default, a local noise reduction filter is applied, modulated by 
``-b`` and ``-bn`` options respectively. Other filers may be set using ``-r``
and ``-rn``:

1. ``mean_below_mean`` - Remove all points below the mean of all points below the mean of all unfiltered points of this scan
2. ``median`` - Remove all points below the median intensity of this scan
3. ``one_percent_of_max`` - Remove all points with intensity less than 1% of the maximum intensity point of this scan
4. ``fticr_baseline`` - Apply the same background reduction algorithm used by ``-b`` and ``-bn``
5. ``savitsky_golay`` - Apply Savtisky-Golay smoothing on the intensities of this scan


Output Information
------------------

The resulting mzML file from this tool attempts to preserve as much metadata as possible
from the source data file, and records its own metadata in the appropriate sections of
the document.

Each scan has a standard set of ``cvParam`` entries covering scan polarity,
peak mode, and MS level. In addition to the normal ``m/z array`` and ``intensity array``
entries, each scan also includes the standardized ``charge array``, as well as two non-standard
arrays, ``deconvolution score array`` and ``isotopic envelopes array``. The ``deconvolution score array``
is just the result of the goodness-of-fit function used to evaluate the isotopic envelopes resulting
in the reported peaks. The ``isotopic envelopes array`` is more complex, as it encodes the set of isotopic
peaks used to fit each reported peak, and does not have a one-to-one relationship with other arrays.

To unpack the ``isotopic envelopes array`` after decoding, the we use the following logic:

.. code-block:: python
    :linenos:

    def decode_envelopes(array):
        '''
        Arguments
        ---------
        array: float32 array
        '''
        envelope_list = []
        current_envelope = []
        i = 0
        n = len(array)
        while i < n:
            # fetch the next two values
            mz = array[i]
            intensity = array[i + 1]
            i += 2

            # if both numbers are zero, this denotes the beginning
            # of a new envelope
            if mz == 0 and intensity == 0:
                if current_envelope is not None:
                    if current_envelope:
                        envelope_list.append(Envelope(current_envelope))
                    current_envelope = []
            # otherwise add the current point to the existing envelope
            else:
                current_envelope.append(EnvelopePair(mz, intensity))
        envelope_list.append(Envelope(current_envelope))
        return envelope_list

Bibliography
------------

.. [Senko1995]
    Senko, M. W., Beu, S. C., & McLafferty, F. W. (1995). Determination of
    monoisotopic masses and ion populations for large biomolecules from resolved
    isotopic distributions.
    Journal of the American Society for Mass Spectrometry, 6(4), 229–233.
    https://doi.org/10.1016/1044-0305(95)00017-8
.. [Martens2011]
    Martens, L., Chambers, M., Sturm, M., Kessner, D., Levander, F., Shofstahl, J.,
    … Deutsch, E. W. (2011). mzML--a community standard for mass spectrometry data.
    Molecular & Cellular Proteomics : MCP, 10(1), R110.000133.
    https://doi.org/10.1074/mcp.R110.000133
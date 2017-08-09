Raw Data Deconvolution
======================


Convert raw mass spectral data files into deisotoped neutral mass peak lists
written to a new **mzML** [Martens2011]_ file. For tandem mass spectra,
recalculate precursor ion monoisotopic peaks.

This task is computationally intensive, and uses several collaborative processes
to share the work.

.. click:: glycan_profiling.cli.mzml:preprocess
	:prog: glycresoft mzml preprocess


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
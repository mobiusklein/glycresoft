Glycopeptide Analysis Tutorial
==============================

This tutorial will cover the steps involved in analyzing glycopeptide
LC-MS/MS data.

You can download the raw data we will analyze from `20150710_3um_AGP_001.mzML.gz <https://www.dropbox.com/s/lm0uc5q31aaju8s/20150710_3um_AGP_001.mzML.gz?dl=0>`_. Please download it and decompress it.

.. code-block:: bash
	:caption: Deconvolution of Glycopeptide LC-MS/MS

	$ glycresoft mzml preprocess -p 6 -v -a glycopeptide -an peptide 20150710_3um_AGP_001.mzML\
		 20150710_3um_AGP_001.preprocessed.mzML

This will deconvolute the LC-MS run, using six worker processes. This may take a significant
amount of time, on the order of one to two hours.

Meanwhile, we can begin setting up the hypothesis. This sample contains predominantly AGP
glycopeptides, so we can start by downloading the AGP protein sequences from UniProt:

.. code-block:: bash
	
	$ echo P19652 >> accession.txt
	$ echo P02763 >> accession.txt
	$ glycresoft tools download-uniprot -i accession.txt -o agp.fa


Copy the following text into a file "combinatorial-rules.txt"

.. code-block:: text
	:caption: Glycan Combinatorial Rules

	Hex 3 10
	HexNAc 2 9
	Fuc 0 5
	Neu5Ac 0 4

	Fuc < HexNAc
	HexNAc > NeuAc + 1

Next, we'll build the glycan hypothesis

.. code-block:: bash
	:caption: Build glycan hypothesis

	$ glycresoft build-hypothesis glycan-combinatorial combinatorial-rules.txt glycans.db


Now, we'll build the glycopeptide hypothesis using these glycans and the protein
FASTA we downloaded earlier

.. code-block:: bash
	:caption: Build glycopeptide hypothesis

	$ glycresoft build-hypothesis glycopeptide-fa -g glycans.db -s hypothesis -G 1\
	        -u 1 -e trypsin -m 1 -c "Carbamidomethyl (C)" -v "Deamidation (N)"\
	        -v "Pyro-glu from Q (Q@N-term)" -p 4 -n "Alpha-1-acid Glycopeptide Hypothesis"\
	        agp.fa fasta-agp.db

This task should take a few minutes at most.


Once both the glycopeptide hypothesis is built and the sample is deconvoluted, we can
run the database search:

.. code-block:: bash
	:caption: Database search process

	$ glycresoft analyze search-glycopeptide fasta-agp.db 20150710_3um_AGP_001.preprocessed.mzML 1\
         -o agp-glycopepitdes-20150710_3um_AGP_001.db -p 5


The search process should take between 2 and 10 minutes.

Once the database search process has completed, we can export the search results in CSV format
for downstream analysis.

.. code-block:: bash
	:caption: CSV export

	$ glycresoft export glycopeptide-identification agp-glycopepitdes-20150710_3um_AGP_001.db 1\
	  -o agp-glycopeptides.csv


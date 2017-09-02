Building a Glycopeptide Hypothesis from a FASTA File
====================================================

The simplest way to build a glycopeptide database is to start with a
list of theoretical glycoproteins in a FASTA file and perform a simple
*in-silico* digest with one or more proteases, apply a set of ``modification rules``,
and combine the resulting peptides with a glycan hypothesis to produce
glycopeptides.

.. warning::

    Glycan compositions read from the associated glycan hypothesis must be
    classified as N-glycans, O-glycans, or GAG-Linkers in order for the algorithm
    to select the appropriate glycosite assignment algorithm. Glycan compositions
    lacking a classification **will not** be considered.

The build process is computationally intensive, as such this tool will spawn
several worker processes to share the load.


.. click:: glycan_profiling.cli.build_db:glycopeptide_fa
    :prog: glycresoft build-hypothesis glycopeptide-fa


Usage Example
-------------

Below is a basic example of how to use this tool

.. code-block:: bash
    
    # We'll build a simple glycopeptide hypothesis from just two glycoproteins, the
    # isoforms of Alpha-1-acid glycoprotein.
    $ cat agp.fa
    >sp|P02763|A1AG1_HUMAN Alpha-1-acid glycoprotein 1 OS=Homo sapiens GN=ORM1 PE=1 SV=1
    MALSWVLTVLSLLPLLEAQIPLCANLVPVPITNATLDQITGKWFYIASAFRNEEYNKSVQ
    EIQATFFYFTPNKTEDTIFLREYQTRQDQCIYNTTYLNVQRENGTISRYVGGQEHFAHLL
    ILRDTKTYMLAFDVNDEKNWGLSVYADKPETTKEQLGEFYEALDCLRIPKSDVVYTDWKK
    DKCEPLEKQHEKERKQEEGES

    >sp|P19652|A1AG2_HUMAN Alpha-1-acid glycoprotein 2 OS=Homo sapiens GN=ORM2 PE=1 SV=2
    MALSWVLTVLSLLPLLEAQIPLCANLVPVPITNATLDRITGKWFYIASAFRNEEYNKSVQ
    EIQATFFYFTPNKTEDTIFLREYQTRQNQCFYNSSYLNVQRENGTVSRYEGGREHVAHLL
    FLRDTKTLMFGSYLDDEKNWGLSFYADKPETTKEQLGEFYEALDCLCIPRSDVMYTDWKK
    DKCEPLEKQHEKERKQEEGES

    # This hypothesis will include a combinatorial glycan composition hypothesis
    # defined by these rules
    $ cat combinatorial-rules.txt
    Hex 3 12
    HexNAc 2 10
    Fuc 0 5
    Neu5Ac 0 4

    Fuc < HexNAc
    HexNAc > NeuAc + 1

    # We'll permit the following additional options:
    #  -u 1 means that only one glycosylation site will ever be occupied
    #  -e trypsin means that the proteins will be *in-silico* digested with trypsin
    #  -m 1 means that we will only permit one missed cleavage by the protease
    #  -c "Carbamidomethyl (C)" means we will have a constant modification "Carbamidomethyl"
    #   on all Cysteine residues
    #  -v "Deamidation (N)" means we will have a variable modification "Deamidation" on any
    #   Asparagine residue
    #  -v "Pyro-glu from Q (Q@N-term)" means that we will have a variable modification "Pyro-glu from Q"
    #   on any Glutamine residue on the N-terminus of the peptide sequence
    #  -p 4 means that this task will spawn four worker processes to share the work between
    $ glycresoft build-hypothesis glycopeptide-fa -g combinatorial-rules.txt -s combinatorial\
        -u 1 -e trypsin -m 1 -c "Carbamidomethyl (C)" -v "Deamidation (N)"\
        -v "Pyro-glu from Q (Q@N-term)" -p 4 -n "Alpha-1-acid Glycopeptide Hypothesis"\
        agp.fa fasta-glycopeptides.db

    2017-08-31 02:50:08.610441 Begin Combinatorial Glycan Hypothesis Serializer
    {'derivatization': None,
     'engine': Engine(sqlite:///fasta-glycopeptides.db),
     'glycan_file': 'combinatorial-rules.txt',
     'loader': None,
     'reduction': None,
     'start_time': datetime.datetime(2017, 8, 31, 2, 50, 8, 608881),
     'status': 'started',
     'transformer': None,
     'uuid': '93604f0929794e2bab9df23e29118b09'}
    2017-08-31 02:50:08.628096 Generating Glycan Compositions from Symbolic Rules for GlycanHypothesis(id=1, name=GlycanHypothesis-93604f0929794e2bab9df23e29118b09)
    2017-08-31 02:50:15.188556 1000 glycan compositions created
    2017-08-31 02:50:20.824145 Generated 1900 glycan compositions
    2017-08-31 02:50:20.835196 Hypothesis Completed
    2017-08-31 02:50:20.835577 End Combinatorial Glycan Hypothesis Serializer
    2017-08-31 02:50:20.835661 Started at 2017-08-31 02:50:08.608881.
    Ended at 2017-08-31 02:50:20.835265.
    Total time elapsed: 0:00:12.226384
    CombinatorialGlycanHypothesisSerializer completed successfully.
    2017-08-31 02:50:20.903601 Begin Multiple Process Fasta Glycopeptide Hypothesis Serializer
    {'constant_modifications': [Carbamidomethyl:57.021464],
     'engine': Engine(sqlite:///fasta-glycopeptides.db),
     'fasta_file': 'agp.fa',
     'max_glycosylation_events': 1,
     'max_missed_cleavages': 1,
     'n_processes': 4,
     'protease': 'trypsin',
     'start_time': datetime.datetime(2017, 8, 31, 2, 50, 20, 899793),
     'status': 'started',
     'total_glycan_combination_count': -1,
     'uuid': '1e6d202801ce4417b3b153ac84732401',
     'variable_modifications': [Deamidated:0.984016, Gln->pyro-Glu:-17.026549]}
    2017-08-31 02:50:20.903709 Extracting Proteins
    2017-08-31 02:50:20.927056 Digesting Proteins
    2017-08-31 02:50:28.965977 205 Base Peptides Produced
    2017-08-31 02:50:28.966163 Begin Applying Protein Annotations
    2017-08-31 02:50:29.726805 ... Extracting Best Peptides
    2017-08-31 02:50:29.753947 ... Building Mask
    2017-08-31 02:50:29.755910 ... Removing Duplicates
    2017-08-31 02:50:29.765368 ... Complete
    2017-08-31 02:50:29.768705 Combinating Glycans
    2017-08-31 02:50:30.766838 ... Building combinations for Hypothesis 1
    2017-08-31 02:50:33.867166 1900 Glycan Combinations Constructed.
    2017-08-31 02:50:33.867282 Building Glycopeptides
    2017-08-31 02:50:33.895503 Begin Creation. Dropping Indices
    2017-08-31 02:50:33.939597 ... Processing Glycan Combinations 0-1900 (100.00%)
    2017-08-31 02:50:34.021121 ... Dealt Peptides 0-11 4.60%
    2017-08-31 02:50:34.021361 ... Dealt Peptides 11-22 9.21%
    2017-08-31 02:50:34.021429 ... Dealt Peptides 22-33 13.81%
    2017-08-31 02:50:34.021495 ... Dealt Peptides 33-44 18.41%
    2017-08-31 02:50:34.021560 ... Dealt Peptides 44-55 23.01%
    2017-08-31 02:50:34.021624 ... Dealt Peptides 55-66 27.62%
    2017-08-31 02:50:34.021781 ... Dealt Peptides 66-77 32.22%
    2017-08-31 02:50:34.021890 ... Dealt Peptides 77-88 36.82%
    2017-08-31 02:50:34.022035 ... Dealt Peptides 88-99 41.42%
    2017-08-31 02:50:34.022238 ... Dealt Peptides 99-110 46.03%
    2017-08-31 02:50:34.340675 ... Dealt Peptides 110-121 50.63%
    2017-08-31 02:50:34.367891 ... Dealt Peptides 121-132 55.23%
    2017-08-31 02:50:34.376397 ... Dealt Peptides 132-143 59.83%
    2017-08-31 02:50:34.410928 ... Dealt Peptides 143-154 64.44%
    2017-08-31 02:50:46.488753 ... Dealt Peptides 154-165 69.04%
    2017-08-31 02:50:48.904966 ... Dealt Peptides 165-176 73.64%
    2017-08-31 02:50:52.744931 ... Dealt Peptides 176-187 78.24%
    2017-08-31 02:50:52.765699 ... Dealt Peptides 187-198 82.85%
    2017-08-31 02:50:52.806290 ... Dealt Peptides 198-209 87.45%
    2017-08-31 02:50:59.408814 ... Dealt Peptides 209-220 92.05%
    2017-08-31 02:51:08.196088 ... Dealt Peptides 220-231 96.65%
    2017-08-31 02:51:11.241885 ... Dealt Peptides 231-239 100.00%
    2017-08-31 02:51:11.241991 ... All Peptides Dealt
    2017-08-31 02:51:33.927773 ... 130001 Glycopeptides Created
    2017-08-31 02:52:10.984681 Process 3560 completed. (41 peptides, 57000 glycopeptides)
    2017-08-31 02:52:11.928619 Process 3556 completed. (66 peptides, 55100 glycopeptides)
    2017-08-31 02:52:12.924645 Process 3558 completed. (66 peptides, 57000 glycopeptides)
    2017-08-31 02:52:17.504664 Process 3562 completed. (66 peptides, 62700 glycopeptides)
    2017-08-31 02:52:19.507377 Joining Process 3556 (False)
    2017-08-31 02:52:19.507669 Joining Process 3558 (False)
    2017-08-31 02:52:19.507839 Joining Process 3560 (False)
    2017-08-31 02:52:19.507991 Joining Process 3562 (False)
    2017-08-31 02:52:19.508170 All Work Done. Rebuilding Indices
    2017-08-31 02:52:21.342381 Analyzing Indices
    2017-08-31 02:52:21.605914 Done Analyzing Indices
    2017-08-31 02:52:21.645477 Generated 231800 glycopeptides
    2017-08-31 02:52:21.645579 Done
    2017-08-31 02:52:21.659734 Hypothesis Completed
    2017-08-31 02:52:21.660019 End Multiple Process Fasta Glycopeptide Hypothesis Serializer
    2017-08-31 02:52:21.660143 Started at 2017-08-31 02:50:20.899793.
    Ended at 2017-08-31 02:52:21.659825.
    Total time elapsed: 0:02:00.760032
    MultipleProcessFastaGlycopeptideHypothesisSerializer completed successfully.


If you instead wish to use an existing glycan hypothesis instead
of creating a new one, you can modify the instructions above:

.. code-block:: bash

    $ glycresoft build-hypothesis glycan-combinatorial rules-file.txt combinatorial-database -n "Combinatorial Human N-Glycans"
    ...

    $ glycresoft build-hypothesis glycopeptide-fa -g combinatorial-database.db -s hypothesis -G 1\
        -u 1 -e trypsin -m 1 -c "Carbamidomethyl (C)" -v "Deamidation (N)"\
        -v "Pyro-glu from Q (Q@N-term)" -p 4 -n "Alpha-1-acid Glycopeptide Hypothesis"\
        agp.fa fasta-glycopeptides.db
    ...

The primary difference here is that the value of the ``-g`` option is the path to the source
database's path (or connection URI), ``-s`` indicates that the source is a hypothesis, and
the new ``-G`` option identifies the hypothesis to use, as a single database may contain many
hypotheses. The value of ``-G`` can be the ID of the hypothesis or the hypothesis's name (in this case
"Combinatorial Human N-Glycans").


Supported Proteases
~~~~~~~~~~~~~~~~~~~

.. exec::

    from glycopeptidepy.enzyme import expasy_rules
    from rst_table import as_rest_table

    rows = [
        ("Enzyme Name", "Recognized Pattern")
    ]

    for name, pattern in expasy_rules.items():
        rows.append((name, pattern))
    
    print(as_rest_table(rows))


Supported Modification Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:title-reference:`glycresoft` supports the full range of `UNIMOD <http://www.unimod.org/modifications_list.php?>`_
modification rules as well as some common alternative namings.

To be more specific, you are able to override modification targets when you
specify modification names by passing the permitted target rules enclosed in
parentheses following the modification name. For example "Deamidation (N)" will
only target Asparagine residues, unlike the plain "Deamidation" rule which will
target both Asparagine and Glutamine.

For more information about supported post-translational modifications, please
see `Peptide Modifications <todo>`_


UniProt Integration
~~~~~~~~~~~~~~~~~~~

todo

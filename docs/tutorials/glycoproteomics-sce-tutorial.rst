Multipart Score Glycopeptide Search Tutorial
============================================

This tutorial will cover the steps involved in analyzing glycopeptide
LC-MS/MS data acquired with stepped collision energy.


Preparing the Data
~~~~~~~~~~~~~~~~~~

To begin, please download the five Thermo .raw files from `PXD005413 <https://www.ebi.ac.uk/pride/archive/projects/PXD005413>`_:

.. code-block:: bash

    # using wget and the PRIDE FTP location programmatically may work
    $ for i in 1 2 3 4 5; do
        wget ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2017/09/PXD005413/MouseHeart-Z-T-${i}.raw
      done

On Windows, you can directly process these files in the next step, but for other platforms, please convert them to
mzML without peak picking. Then run the MS deconvolution tool ``glycresoft mzml preprocess`` :ref:`CLI Documentation <mzml_preprocess>`.

.. note:: Replace the ``$CPUS`` variable with the number of cores you want the program to use per command.

.. code-block:: bash

    $ for i in 1 2 3 4 5; do
        glycresoft -l preprocess-${i}.log mzml preprocess \
            -p $CPUS \
            -v \
            -b 0 \
            -g 1 \
            -a glycopeptide \
            -c 12 \
            -an peptide \
            -tn 10 \
            MouseHeart-Z-T-${i}.$ext MouseHeart-Z-T-${i}.deconv.mzML
     done

Building the Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~

Meanwhile, we need to prepare the database GlycReSoft will search. Please download the UniProt Mouse reference proteome and the glycan list:

.. code-block:: bash

    wget "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28proteome%3AUP000000589%29+AND+reviewed%3Dtrue%29" -O UP000000589_mouse_reference_proteome_sp_only.fa
    wget --no-check-certificate "https://raw.githubusercontent.com/mobiusklein/glycresoft/master/docs/tutorials/combined_mouse_nglycans.txt" -O combined_mouse_nglycans.txt


The multi-part scoring scheme requires a separate target database and decoy database, and does not fully materialize the crossproduct of peptides and glycans,
dynamically computing them as needed at run time. Therefore we need to build the database twice, once for the targets and once for the decoys with the ``--reverse``
flag set, and we pass the ``-F / --not-full-crossproduct`` flag to both database build commands to prevent the crossproduct from being generated
consequently taking substantially less time.

We build the target database:

.. code-block:: bash

    glycresoft -l build-target-db.log build-hypothesis glycopeptide-fa -p $CPUS \
        -g combined_mouse_nglycans.txt -s text \
        -u 1 \
        -c "Carbamidomethyl (C)" \
        -v "Oxidation (M)" \
        -m 2 -e trypsin \
        -C \
        -F \
        UP000000589_mouse_reference_proteome_sp_only.fa  UP000000589_mouse_sp_only_glycoproteome.db

    export TARGET_DB=UP000000589_mouse_sp_only_glycoproteome.db

and decoy database:

.. code-block:: bash

    glycresoft -l build-decoy-db.log build-hypothesis glycopeptide-fa -p $CPUS \
        -g combined_mouse_nglycans.txt -s text \
        -u 1 \
        -c "Carbamidomethyl (C)" \
        -v "Oxidation (M)" \
        -m 2 -e trypsin \
        -C \
        -F \
        --reverse \
        UP000000589_mouse_reference_proteome_sp_only.fa  UP000000589_mouse_sp_only_glycoproteome.decoy.db

    export DECOY_DB=UP000000589_mouse_sp_only_glycoproteome.decoy.db


Searching the Data
~~~~~~~~~~~~~~~~~~

Once the databases are built and the spectra have been preprocessed, we can run the search step (:ref:`CLI Documentation <search-glycopeptide-multipart>`).

This will write both the complete results recorded a SQLite file as well as a more easily readable CSV file of the glycopeptide spectrum matches.

.. code-block:: bash

    $ for i in 1 2 3 4 5; do
        glycresoft -l search-${i}.log analyze search-glycopeptide-multipart \
            -p $CPUS \
            -w 500 \
            -m 5e-6 \
            -mn 2e-5 \
            -s log_intensity_v3 \
            -o Mouse-Heart-${i}.search.db \
            -M \
            -a Ammonium 2 \
            --export psm-csv \
            $TARGET_DB $DECOY_DB \
            MouseHeart-Z-T-${i}.deconv.mzML
     done

Build a Glycosite Smoothing Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a site-specific glycome network smoothing model, we need to build a graph first.
(:ref:`CLI Documentation for build-network <build-glycan-graph>`, :ref:`CLI Documentation for add-prebuilt-neighborhoods <add-predefined-neighborhood-glycan-graph>`)

.. code-block:: bash

    $ glycresoft build-hypothesis glycan-network build-network $TARGET_DB 1 -o mouse_glycan_network.txt
    $ glycresoft build-hypothesis glycan-network add-prebuilt-neighborhoods -n mammalian-n-glycan \
        -i mouse_glycan_network.txt \
        -o mouse_glycan_network.txt

Then, we can run the glycosite smoothing model building workflow (:ref:`CLI Documentation <build-glycosite-model>`):

.. code-block:: bash

    $ glycresoft analyze fit-glycoproteome-smoothing-model \
        -p $CPUS
        -i Mouse-Heart-1.search.db 1 \
        -i Mouse-Heart-2.search.db 1 \
        -i Mouse-Heart-3.search.db 1 \
        -i Mouse-Heart-4.search.db 1 \
        -i Mouse-Heart-5.search.db 1 \
        -w mouse_glycan_network.txt \
        -q 0.01 \
        -g $TARGET_DB 1 \
        -P $TARGET_DB 1 \
        -o mouse-heart-heart-glycosite-models.json

Then we can re-analyze the dataset with the smoothing model:

.. code-block:: bash

    $ for i in 1 2 3 4 5; do
        glycresoft -l search-smoothed-${i}.log analyze search-glycopeptide-multipart \
            -p $CPUS \
            -w 500 \
            -m 5e-6 \
            -mn 2e-5 \
            -s log_intensity_v3 \
            -o Mouse-Heart-${i}.search-smoothed.db \
            -M \
            -a Ammonium 2 \
            --export psm-csv \
            -S mouse-heart-heart-glycosite-models.json \
            $TARGET_DB $DECOY_DB \
            MouseHeart-Z-T-${i}.deconv.mzML
     done


Build a Fragmentation Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to export annotated MGF files from the identification results.

.. code-block:: bash

    $ for i in 1 2 3 4 5; do
        glycresoft -l export-training-mgf-${i}.log glycopeptide-training-mgf \
            -o Mouse-Heart-${i}.training.mgf \
            Mouse-Heart-${i}.search.db 1

Then, ensure `glycopeptide_feature_learning <https://github.com/mobiusklein/glycopeptide_feature_learning>`_
is installed. Now we can fit the fragmentation model:

.. code-block:: bash

    $ DATAFILES=`ls Mouse-Heart-*.training.mgf`
    $ MODEL_NAME=mouse-heart-fragmodel
    $ # Do the model training
    $ glycopeptide-feature-learning fit-model -t 20 $DATAFILES -o ${MODEL_NAME}.json
    $ # Convert the complete model into something easier for Python to read
    $ glycopeptide-feature-learning compile-model ${MODEL_NAME}.json ${MODEL_NAME}.pkl
    $ # Evaluate the model fit
    $ glycopeptide-feature-learning calculate-correlation -t 20 $DATAFILES ./correlation.${MODEL_NAME}.pkl  ${MODEL_NAME}.pkl
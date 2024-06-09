Indexed Glycopeptide Search Tutorial
====================================

This tutorial will cover the steps involved in analyzing glycopeptide
LC-MS/MS data acquired with stepped collision energy.

To begin, please download the five Thermo .raw files from `PXD005413 <https://www.ebi.ac.uk/pride/archive/projects/PXD005413>`_:

.. code-block:: bash

    # using wget and the PRIDE FTP location programmatically may work
    $ for i in 1 2 3 4 5; do
        wget ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2017/09/PXD005413/MouseHeart-Z-T-${i}.raw
      done

On Windows, you can directly process these files in the next step, but for other platforms, please convert them to
mzML without peak picking. Then run the MS deconvolution tool ``ms-deisotope``, which is part of the ``ms_deisotope`` package.

Replace the ``$CPUS`` variable with the number of cores oyu want the program to use per file.

.. code-block:: bash

    $ for i in 1 2 3 4 5; do
        ms-deisotope -p $CPUS -v -b 0 -g 1 -a glycopeptide -c 12 -an peptide -tn 10 MouseHeart-Z-T-${i}.$ext MouseHeart-Z-T-${i}.deconv.mzML
     done

Meanwhile, we need to prepare the database GlycReSoft will search. Please download the UniProt Mouse reference proteome:


.. code-block:: bash

    wget "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28proteome%3AUP000000589%29+AND+reviewed%3Dtrue%29" -O UP000000589_mouse_reference_proteome_sp_only.fa


#!/usr/bin/env bash

glycresoft build-hypothesis glycopeptide-fa -c "Carbamidomethyl (C)" \
    -v "Pyro-glu from Q (Q@N-term)" -v "Oxidation (M)" \
    -m 1 -e trypsin -g "./agp_glycans.txt" "./agp.fa" agp.db


glycresoft build-hypothesis glycopeptide-fa -c "Carbamidomethyl (C)" \
    -v "Pyro-glu from Q (Q@N-term)" -v "Oxidation (M)" \
    -m 1 -e trypsin -g "./agp_glycans.txt" -F "./agp.fa" agp_indexed.db


glycresoft build-hypothesis glycopeptide-fa -c "Carbamidomethyl (C)" \
    -v "Pyro-glu from Q (Q@N-term)" -v "Oxidation (M)" \
    -m 1 -e trypsin -g "./agp_glycans.txt" -F -R "./agp.fa" agp_indexed_decoy.db

glycresoft analyze search-glycopeptide -o "./classic_agp_search.db" \
    "./agp.db" "./20150710_3um_AGP_001_29_30.preprocessed.mzML" 1

glycresoft analyze search-glycopeptide -o "./classic_agp_search_empty.db" \
    "./agp.db" "./AGP_Glycomics_20150930_06.deconvoluted.mzML" 1

glycresoft analyze search-glycopeptide-multipart -o "./indexed_agp_search.db" -M \
    "./agp_indexed.db" "./agp_indexed_decoy.db" "./20150710_3um_AGP_001_29_30.preprocessed.mzML"

glycresoft analyze search-glycopeptide-multipart -o "./indexed_agp_search_empty.db" -M \
    "./agp_indexed.db" "./agp_indexed_decoy.db" "./AGP_Glycomics_20150930_06.deconvoluted.mzML"
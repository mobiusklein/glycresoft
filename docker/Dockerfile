FROM mobiusklein/ms_deisotope

RUN pip install sqlalchemy dill jinja2 rdflib requests SPARQLWrapper

RUN pip install glypy==0.12.6
RUN pip install git+https://github.com/mobiusklein/glycopeptidepy
RUN pip install git+https://github.com/mobiusklein/glycresoft
RUN pip install git+https://github.com/mobiusklein/msms_feature_learning

# glycresoft installed and accessible
CMD ["/bin/bash"]
FROM mobiusklein/ms_deisotope

ARG GLYCRESOFT_VERSION=master

RUN pip install sqlalchemy dill jinja2 rdflib requests SPARQLWrapper

RUN pip install glypy
RUN pip install git+https://github.com/mobiusklein/glycopeptidepy

RUN pip install git+https://github.com/mobiusklein/glycresoft@${GLYCRESOFT_VERSION}
RUN pip install git+https://github.com/mobiusklein/msms_feature_learning

# glycresoft installed and accessible
CMD ["/bin/bash"]
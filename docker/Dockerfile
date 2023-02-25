FROM mobiusklein/ms_deisotope

ARG GLYCRESOFT_VERSION=master

RUN pip install sqlalchemy dill jinja2 rdflib requests SPARQLWrapper scikit-learn

RUN pip install glypy
RUN pip install glycopeptidepy

RUN pip install git+https://github.com/mobiusklein/glycresoft@${GLYCRESOFT_VERSION} --no-build-isolation
RUN pip install git+https://github.com/mobiusklein/msms_feature_learning --no-build-isolation

# glycresoft installed and accessible
CMD ["/bin/bash"]
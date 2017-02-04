FROM openml/jupyter-python:0.1.2

RUN bash --login -c "conda install -qy pillow graphviz && pip install graphviz"
RUN bash --login -c "pip install git+https://github.com/renatopp/liac-arff@master"
RUN bash --login -c "pip install git+https://github.com/openml/openml-python.git@develop"

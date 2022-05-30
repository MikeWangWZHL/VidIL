FROM ruox/blip_environment_a100

ADD install_pkg_minimum.sh install_pkg_minimum.sh
ADD requirements.txt requirements.txt
ADD apex/ apex/

RUN bash install_pkg_minimum.sh
RUN pip install -U sentence-transformers
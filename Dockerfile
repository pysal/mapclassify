# Ubuntu Bionic 18.04 at Jan 26'19
FROM jupyter/minimal-notebook:87210526f381

MAINTAINER Serge Rey <sjsrey@gmail.com>

# https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/Dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root

#--- Utilities ---#

RUN apt-get update \
  && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get install -y libgeos-dev

ADD install_deps_py.sh $HOME/install_deps_py.sh

USER root
RUN chmod +x $HOME/install_deps_py.sh
RUN sed -i -e 's/\r$//' $HOME/install_deps_py.sh
RUN ["/bin/bash", "-c", "$HOME/install_deps_py.sh"]
RUN rm /home/jovyan/install_deps_py.sh 

# Switch back to user to avoid accidental container runs as root
USER $NB_UID

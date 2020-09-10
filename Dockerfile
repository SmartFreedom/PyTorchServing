FROM nvidia/cuda:10.1-base-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir -p /opt/entrypoint
WORKDIR /opt/entrypoint

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /opt/entrypoint
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl 'https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh' --output $HOME/conda.sh \
 && chmod +x $HOME/conda.sh \
 && $HOME/conda.sh -b -p $HOME/conda \
 && rm $HOME/conda.sh
ENV PATH=$HOME/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN $HOME/conda/bin/conda create -y --name py36 python=3.6.9 \
 && $HOME/conda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=$HOME/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN $HOME/conda/bin/conda install conda-build=3.18.9=py36_3 \
 && $HOME/conda/bin/conda clean -ya

# Install wget & make
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    wget \
    build-essential  \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Redis
RUN cd $HOME && wget http://download.redis.io/redis-stable.tar.gz \
 && tar xvzf redis-stable.tar.gz \
 && cd redis-stable \
 && make && sudo make install \
 && cd $HOME \
 && rm redis-stable.tar.gz

# Copy repository and data
COPY --chown=user . /opt/entrypoint/
RUN chmod -R 765 /opt/entrypoint/

# Create an environment
RUN conda env create -f /opt/entrypoint/environment.yml
RUN echo "source activate $(head -1 /opt/entrypoint/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /opt/entrypoint/environment.yml | cut -d' ' -f2)/bin:$PATH

ENV DEBIAN_FRONTEND teletype
ENV REDIS_DB_V=$REDIS_DB_V

# Set the default command to launch redis & torch server
EXPOSE 8888 6006 22 9358 9769 6379
# ENTRYPOINT /bin/bash

ENTRYPOINT /bin/bash init.sh

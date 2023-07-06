# Created by: George Corrêa de Araújo (george.gcac@gmail.com)
# ==================================================================

# FROM python:latest
FROM python:3.11

ARG GROUPID=901
ARG GROUPNAME=cleaner
ARG USERID=901
ARG USERNAME=user

# Environment variables

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install --upgrade" && \

# ==================================================================
# Create a system group with name deeplearning and id 901 to avoid
#    conflict with existing uids on the host system
# Create a system user with id 901 that belongs to group deeplearning
# ------------------------------------------------------------------

    groupadd -r $GROUPNAME -g $GROUPID && \
    # useradd -u $USERID -r -g $GROUPNAME $USERNAME && \
    useradd -u $USERID -m -g $GROUPNAME $USERNAME && \

# ==================================================================
# libraries via apt-get
# ------------------------------------------------------------------

    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        curl \
        python3-enchant \
        locales \
        wget && \

# ==================================================================
# python libraries via pip
# ------------------------------------------------------------------

    $PIP_INSTALL \
        pip \
        wheel && \
    $PIP_INSTALL \
        colorama \
        ftfy \
        # inflect \
        ipdb \
        ipython \
        nltk \
        pandas \
        pyarrow \
        pyenchant \
        pypdfium2 \
        tqdm && \

    # until these are merged into official inflect, use this fork
    # https://github.com/jaraco/inflect/pull/167
    # https://github.com/jaraco/inflect/pull/168
    $PIP_INSTALL \
        # git+https://github.com/george-gca/inflect@fix_s_plural_noun && \
        git+https://github.com/george-gca/inflect && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sed -i -e 's/# pt_BR.UTF-8 UTF-8/pt_BR.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LC_ALL pt_BR.UTF-8

USER $USERNAME

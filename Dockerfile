FROM mambaorg/micromamba:2.3.3

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYVISTA_OFF_SCREEN=true \
    UV_SYSTEM_PYTHON=1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxrender1 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /workspace

RUN micromamba install -y -n base -c conda-forge \
        python=3.12 \
        fenics-dolfinx \
        mpich \
        pip \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src

RUN python -m pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --no-cache -e ".[cfd,fem,viz,dev]"

CMD ["/bin/bash"]

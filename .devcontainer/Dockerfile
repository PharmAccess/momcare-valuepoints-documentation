# TODO: update to latest again when connectorx 0.3.4 released
FROM condaforge/mambaforge:24.3.0-0
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive


# librsvg2-bin is to allow SVG conversion when rendering a PDF file
# (will install the rsvg-view binary)
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    pandoc \
    pandoc-citeproc \
    curl \
    gdebi-core \
    librsvg2-bin \
    wget \
    dpkg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Quarto v1.6.42
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb && \
    dpkg -i quarto-1.6.42-linux-amd64.deb || apt-get install -f -y && \
    rm quarto-1.6.42-linux-amd64.deb

# libraries
RUN mamba install --yes \
    "python>=3.10" \
    openjdk=17 \
    # cleanup
    && mamba clean -tipy

RUN pip install \
    dbt-core \
    polars \
    "duckdb==1.0.0" \
    # TODO: update to pathling 8.0.0 once it is released
    pathling==8.0.0.dev0
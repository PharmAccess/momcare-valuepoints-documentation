FROM condaforge/miniforge3:latest
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

# Install Go using the official binary distribution
ENV GO_VERSION=1.20.6
RUN wget https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt
ENV PATH="/usr/local/go/bin:${PATH}"

# libraries
RUN mamba install --yes \
    "python>=3.11" \
    openjdk=17 \
    go-task \
    # cleanup
    && mamba clean -tipy

RUN pip install \
    dbt-core \
    polars \
    duckdb \
    # TODO: update to pathling 8.0.0 once it is released
    pathling==8.0.0.dev0
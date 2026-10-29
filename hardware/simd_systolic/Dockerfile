FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates python3 python3-numpy python3-venv \
    openjdk-17-jdk cmake ninja-build autoconf flex bison help2man perl \
    yosys libqt5charts5 libqt5widgets5 libqt5gui5 tcl-tclreadline \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v5.046 https://github.com/verilator/verilator /opt/verilator \
    && cd /opt/verilator && autoconf && ./configure && make -j8 && make install
RUN curl -fL https://github.com/sbt/sbt/releases/download/v1.9.7/sbt-1.9.7.tgz \
    | tar xz -C /opt && ln -s /opt/sbt/bin/sbt /usr/local/bin/sbt
WORKDIR /workspace/hardware/simd_systolic
COPY . .
RUN python3 scripts/bootstrap.py --physical
CMD ["make", "smoke"]

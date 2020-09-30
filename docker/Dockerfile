FROM hysds/pge-base:v3.0.5
MAINTAINER edunn "Alexander.E.Dunn@jpl.nasa.gov"
LABEL description="Multi-Temporal Anomaly Detection for SAR Earth Observations ARIMA Model"

ARG id
ARG gid

ENV verdi_root="/home/ops/verdi/ops"
ENV pge_root="${verdi_root}/arima-pge"

# Install pip dependencies
RUN echo "===Installing pip dependencies==="
COPY requirements.txt $pge_root/requirements.txt
RUN /opt/conda/bin/pip install -r ${pge_root}/requirements.txt

# Copy in scripts and configuration
COPY --chown=ops:ops . $pge_root
RUN chmod a+x ${pge_root}/Arima.py
ENV PATH=${PATH}:${pge_root}

USER root
# TODO: REMOVE NEXT LINE ONCE LOCAL TESTS ARE COMPLETE
RUN mv ${pge_root}/examples/* ${pge_root}

USER ops
CMD ["Arima.py"]
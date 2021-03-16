FROM continuumio/miniconda

# Add environment file
RUN mkdir /usr/dev/
COPY ./code/environment.yml /usr/dev/

# Set workdir
RUN mkdir /usr/app/
WORKDIR /usr/app/

# Create environment
ENV CONDA_ENV ma-simulation
RUN conda env create -n ${CONDA_ENV} -f /usr/dev/environment.yml
RUN echo "source activate ${CONDA_ENV}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH

# Expose Port
EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
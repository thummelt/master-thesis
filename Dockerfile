FROM continuumio/miniconda

# Add environment file
COPY ./code/environment.yml /usr/src/app/

# Set workdir
RUN mkdir /usr/src/app/src
WORKDIR /usr/src/app/src

# Create environment
ENV CONDA_ENV ma-simulation
RUN conda env create -n ${CONDA_ENV} -f /usr/src/app/environment.yml
RUN echo "source activate ${CONDA_ENV}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV} /bin:$PATH

# Expose Port
EXPOSE 8888

# Add Environment to NB
#RUN conda run -n ${CONDA_ENV} -m ipykernel install --user --name=${CONDA_ENV}

# Conigure Jupyter NB
#ENV TINI_VERSION v0.6.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
#RUN chmod +x /usr/bin/tini
#ENTRYPOINT ["/usr/bin/tini", "--"]


#CMD [ "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" ]
ENTRYPOINT /bin/bash 
FROM continuumio/miniconda

# Add environment file
RUN mkdir /usr/dev/
COPY ./code/environment.yml /usr/dev/

# Set workdir
RUN mkdir /usr/app/
WORKDIR /usr/app/

# Install fonts
COPY ./TIMES.TTF  /usr/share/fonts/truetype/

# refresh system font cache
#RUN fc-cache -f -v

# refresh matplotlib font cache
RUN rm -fr ~/.cache/matplotlib

# Install tex
#RUN apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super

# Create environment
ENV CONDA_ENV ma-simulation
RUN conda config --set ssl_verify no
RUN conda env create -n ${CONDA_ENV} -f /usr/dev/environment.yml
RUN echo "source activate ${CONDA_ENV}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH

# Expose Port
EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.max_buffer_size=32212254720", "--ResourceUseDisplay.track_cpu_percent=True"]
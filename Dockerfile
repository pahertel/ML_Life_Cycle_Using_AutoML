# Set up the operating system and source code for Docker
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install python3 python3-pip python3-dev -y
RUN pip3 -q install pip --upgrade

# Set the working directory in the container
RUN mkdir scr
WORKDIR scr/

# Copy the files to the working directory
COPY Resources ./Resources
COPY 0.classification_automl.ipynb ./
COPY 1.batch_inference.ipynb ./
COPY custom_classification_functions.py ./
COPY requirements.txt ./

# Install Dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888

# Commands to run on container start
# Allow Root is activated and not best practice
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
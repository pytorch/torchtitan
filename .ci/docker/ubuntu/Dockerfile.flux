FROM 308535385114.dkr.ecr.us-east-1.amazonaws.com/torchtitan/torchtitan-ubuntu-20.04-clang12:latest

# Install flux dependencies
COPY flux-requirements.txt /opt/conda/
RUN . /opt/conda/bin/activate py_${PYTHON_VERSION} && \
    pip install -r /opt/conda/flux-requirements.txt && \
    rm /opt/conda/flux-requirements.txt

USER ci-user
CMD ["bash"]

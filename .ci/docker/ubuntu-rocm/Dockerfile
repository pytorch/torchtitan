# base image
FROM rocm/pytorch-nightly:latest

# args
ARG OS_VERSION
ARG CLANG_VERSION
ARG GCC_VERSION
ARG MINICONDA_VERSION
ARG PYTHON_VERSION

# install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r ./requirements.txt

CMD ["bash"]

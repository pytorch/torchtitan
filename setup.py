from setuptools import find_packages, setup

setup(
    name="torchtitan",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    description="Package for training large models using native PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/torchtitan",
)

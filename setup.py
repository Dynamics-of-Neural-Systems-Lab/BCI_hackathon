"""Setup HCI."""
import numpy
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

setup(
    name="hci",
    version="1.0",
    author="Robert Peach",
    author_email="r.peach13@imperial.ac.uk",
    description="""HCI kaggle competition.""",
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "seaborn",
        "dython",
        "audiomentations",
        "natsort",
        "simple_parsing",
        "wandb",
        "loguru",
        "safetensors",
        "einops",
    ],
)

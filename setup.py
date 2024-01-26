# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

install_requires = [
    "hydra-core",
    "rich",
    "pydantic",
    "scikit-learn",
    "pandas",
    "numpy"
]

setup(
    name="llm_osr",
    version="0.0.1",
    description="Python package for an empirical evaluation of llms versus osr",
    author="Alexander Grote",
    author_email="alexandergrote.ag@gmail.com",
    url="https://github.com/alexandergrote/llm_osr",
    packages=find_packages(exclude=("tests", "docs")),
)
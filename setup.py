#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='UvA FACT 2021- Replicating ""',
    version='0.0.0',
    description='Assignment for Fairness, Accountability, Confidentiality and Transparency in AI-University of Amsterdam. The goal of this project is to assess the reproducibility of NeurIPS 2020 paper[Generative causal explanations of black-box classifiers](https: // arxiv.org/abs/2006.13913).',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/shin-ee-chen/UvA_FACT_2021',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

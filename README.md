# UvA_FACT_2021
Assignment for Fairness, Accountability, Confidentiality and Transparency in AI-University of Amsterdam
The goal of this project is to assess the reproducibility of NeurIPS 2020 paper [Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913).

# TO DO: 
    - Clean up code, and write some documentation
    - Introduce structure for including other encoders/decoders
        - Preferably these are all handled using a properly structured arg-parser
    - Check training bottlenecks
        - Latent variable sweep takes too long (2x training) and time increases over epoches
        - For actual training, the approximate computation of the mutual information is likely the bottle-neck
            - Better approximation methods exist
            - More efficient versions of this algorithm can likely be found (e.g. chunked matrix calculations, Cython)
    - Alter sweep code to allow for reproducing other figures

# Students
Xinyi Chen

Qingzhi Hu

Mario Holubar

Ivo Verhoeven

# Requirements

# How to start

# Results
For images visualising CVAE's latent space, see the figures directory. Images generated during training are stored as, Model>Epoch>Variable.

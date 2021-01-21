- [UvA_FACT_2021](#uva_fact_2021)
- [Structure](#structure)
- [Students](#students)
- [Requirements](#requirements)
- [How to start](#how-to-start)
- [Results](#results)
- [Errors in implementation](#errors-in-implementation)

# UvA_FACT_2021
Assignment for Fairness, Accountability, Confidentiality and Transparency in AI-University of Amsterdam
The goal of this project is to assess the reproducibility of NeurIPS 2020 paper [Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913).

# Structure
```
UvA_FACT_2021 
│   └── training scripts are currently kept in root due to issues with Python packaging
├── checkpoints
│   └── model training output, including lightning/tensorboard logs and pretrained weights 
├── datasets
│   ├── raw data sets in named directories
│   ├── mnist.py
│   └──     file that contains code process MNIST, limiting it to specified classes only
├── figures
│   │     all figures produced during training, or otherwise relevant to the project Maybe should be placed in root directory
│   └──   PyTorch (Lightning) code for creating all training all models described 
├── models
│   ├── mnist_cnn.py
│   │       models for classifying the limitied MNIST datasets
│   ├── mnist_cvae.py
│   └──     cvae for explaining the limitied MNIST classifier  
├── utils
│   ├── cvae_latent_visualization.py
│   │       module for generating latent variable sweeps for a cvae
│   ├── information_flow.py
│   │       approximate mutual information computation using OShaugnessy's source code
│   ├── reproducibility.py
│   │       module with some methods for reproducibility, including universal seed setting, model loading, etc.
│   ├── vae_loss.py
│   └──     module for computing variational loss (ELBO, KLD, BPD)
├── experiments
│   └── will contain all code for experimentation using pretrained models
```

# Students
Xinyi Chen

Qingzhi Hu

Mario Holubar

Ivo Verhoeven

# Requirements

# How to start
### For getting Figure 5(ab) in the original pape, please run the following command
```
python plot5ab.py
```
The produced figures would be in the directory figures/Figure5ab.


# Results
For images visualising CVAE's latent space, see the figures directory. Images generated during training are stored as, Model>Epoch>Variable.

# Errors in implementation
Some differences/errors in suggested implementation found in the paper vs. the actual existing Github repository.
    * Paper suggested that ADAM was used for classifier optimizer, actual code used SGD with momentum
        * Also uses learning rate scheduler, although parser is missing gamma coefficient for decay

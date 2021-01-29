- [UvA_FACT_2021](#uva_fact_2021)
- [Structure](#structure)
- [Students](#students)
- [Text-Extension](#text-extension)
  - [Environment](#environment)
  - [Training](#training)

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
└── utils
    ├── cvae_latent_visualization.py
    │       module for generating latent variable sweeps for a cvae
    ├── information_flow.py
    │       approximate mutual information computation using OShaugnessy's source code
    ├── reproducibility.py
    │       module with some methods for reproducibility, including universal seed setting, model loading, etc.
    ├── vae_loss.py
    └──     module for computing variational loss (ELBO, KLD, BPD)
```


# Text-Extension

This repository contains the code necessary for making the extension to classifying and explaining SST movie reviews. 
## Environment
We provide a conda environment called FACT which contains all packages you might need for running the repo. For your own computer, the environment.yml suggests the local packages required. As we do not have local computer with GPU to train all the models, rather we use Lisa environment with GPU provided by the deep learning course at UvA with environment_Lisa.yml which installs the environment dl2020 with CUDA 10.1 support. 

Running on gpu (note all the models are trained on gpu, so you will get error if you try to load the pretrained models on cpu):
- add the following lines in your ".bashrc":
```
module load 2019
module load Miniconda3/4.7.10
```
- run the following command once:
```
conda env create -f environment_Lisa.yml
```
- add the following lines at the beginning of your experiment script (.sh), before running your Python script:
```
source activate dl2020
pip install nltk
```
Follow the detailed description [here](https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/assignment_1/1_mlp_cnn/README.md).


## Training
For training classifier
```
run python sst_classifier_train.py --max_epochs 15
```

For training text-VAE
```
srun python lm_vae_train.py --max_epochs 100 --inner_iter 250 --max_aggressive_epochs 25 --min_scheduler_epoch 15 --warm_up 10 --sample_every 5
```

For fine-tuning GCE (note, requires pre-trained classifier AND text-VAE)
```
run python lm_gce_train.py --max_epochs 15 --batch_size 64 --Nalpha 32 --Nbeta 8 --lr 1e-3 --lamb 1e-3
```

Samples for training, where possible, can be found in the checkpoints directory.

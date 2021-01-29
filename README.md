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
## Environment
We provide a conda environment called FACT which contains all packages you might need for running the repo. For your own computer, the environment.yml suggests the local packages required. As we do not have local computer with GPU to train all the models, rather we use Lisa environment provided by the deep learning course with environment_Lisa.yml which installs the environment FACT with CUDA 10.1 support. 

- add the following lines in your ".bashrc":
```
module load 2019
module load Miniconda3/4.7.10
```
- run the following command once:
```
conda env create -f environment_Lisa.yml
```
- add the following line at the beginning of your experiment script (.sh), before running your Python script:
```
source activate dl2020
```
Follow the detailed description [here](https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/assignment_1/1_mlp_cnn/README.md).

# How to start
### Running one script to get results
We provide a jupyter notebook (.ipynb) for displaying all the results. In case you want to run different parts separately, you can always check the following commands:

### Train on 1/4/9 MNIST dataset:
1. To train CNN classifier:
```
python mnist_classifier_train.py --classes 1 4 9 --max_epochs 30 \
--datasets traditional
 ```

2. To train VAE and generate Figure 13:
```
python mnist_cvae_train.py --classes 1 4 9  --max_steps 8000 \
--batch_size 64 --lr 5e-4 --Nalpha 75 --Nbeta 25 --K 2 --L 2 --lamb 0.1 \
--dataset traditional
 ```

### Train on fashion MNIST dataset:
1. To train CNN classifier:
```
python mnist_classifier_train.py --classes 0 3 4 --max_epochs 50 \
--datasets fashion --log_dir fmnist_cnn
 ```

2. To train VAE and generate Figure 17:
```
python mnist_cvae_train.py --classes 0 3 4  --max_steps 8000 \
--batch_size 32 --lr 1e-4 --Nalpha 100 --Nbeta 25 --K 2 --L 4 --lamb 0.05 \
--dataset fashion --log_dir fmnist_gce --classifier_path fmnist_cnn_034
 ```

### Figure 5(ab) for information flow and removing aspects
For getting Figure 5(ab) in the original pape, please run the following command:
```
python ablation_study.py
```
The produced figures would be in the directory figures/ablation_study/information_flow.png and figures/ablation_study/accuracy_comparison.png



# Results
For images visualising CVAE's latent space, see the figures directory. Images generated during training are stored as, Model>Epoch>Variable.

# Errors in implementation
Some differences/errors in suggested implementation found in the paper vs. the actual existing Github repository.
    * Paper suggested that ADAM was used for classifier optimizer, actual code used SGD with momentum
        * Also uses learning rate scheduler, although parser is missing gamma coefficient for decay

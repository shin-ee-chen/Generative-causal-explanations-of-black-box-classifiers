- [UvA_FACT_2021](#uva_fact_2021)
- [Structure](#structure)
- [Requirements](#requirements)
- [How to start](#how-to-start)
- [Results](#results)

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
│   ├── fashion_mnist.py
│   └──     file that contains code process FMNIST, limiting it to specified classes only
├── figures
│   │     all figures produced during training, or otherwise relevant to the project Maybe should be placed in root directory
│   └──   PyTorch (Lightning) code for creating all training all models described 
├── models
│   ├── mnist_cnn.py
│   │       models for classifying the limitied MNIST datasets
│   ├── mnist_cvae.py
│   └──     cvae for explaining the limitied MNIST classifier  
├── pretrained_models
│   ├── all the pretrained models are here
├── utils
│   ├── cvae_latent_visualization.py
│   │       module for generating latent variable sweeps for a cvae
│   ├── information_flow.py
│   │       approximate mutual information computation using OShaugnessy's source code
│   ├── reproducibility.py
│   │       module with some methods for reproducibility, including universal seed setting, model loading, etc.
│   ├── vae_loss.py
│   └──     module for computing variational loss (ELBO, KLD, BPD)
│   ├── timing.py
│   └──     module for computing the time required for runnign the experiments
├── generate_figures.py
│   └── one script to get all the results.
├── environment_Lisa.yml
│   └── environment file for training on gpu.
├── environment.yml
│   └── environment file for training locally.
├── find_params.py
│   └── Algorithm I (in the paper) implementation.
├── mnist_classifier_train.py
│   └── file for training the classifier (for both mnist and fmnist).
├── mnist_cvae_train.py
│   └── file for training the gce (for both mnist and fmnist).
```



# Requirements
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
- add the following line at the beginning of your experiment script (.sh), before running your Python script:
```
source activate dl2020
```
Follow the detailed description [here](https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/assignment_1/1_mlp_cnn/README.md).

# How to start
### Running one script to get results
We provide one single python file for displaying all the results (including figure 5ab in the original paper for information flow and comparison of accuracies). Run the following command to get all the reproduced figures/results: 
```
python generate_figures.py
```
For training the classifier and gce, you need the following code and it will save the models into pretrained_models:
### Train on 3/8 MNIST dataset:
1. To train CNN classifier:
```
python mnist_classifier_train.py --classes 3 8 --max_epochs 20 \
--datasets traditional
 ```

2. To train VAE and generate Figure 13:
```
python mnist_cvae_train.py --classes 3 8  --max_steps 8000 \
--batch_size 64 --lr 5e-4 --Nalpha 100 --Nbeta 25 --K 1 --L 7 --lamb 0.05 \
--dataset traditional
 ```

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

# Results
All the results are saved in ./figures directory.

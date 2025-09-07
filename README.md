- [UvA_FACT_2021](#uva_fact_2021)
  - [Report Conclusion](#report-conclusion)
- [Structure](#structure)
- [Students](#students)
- [Requirements](#requirements)
  - [Environment](#environment)
- [How to start](#how-to-start)
    - [Running one script to get results](#running-one-script-to-get-results)
    - [Train on 3/8 MNIST dataset:](#train-on-38-mnist-dataset)
    - [Train on 1/4/9 MNIST dataset:](#train-on-149-mnist-dataset)
    - [Train on fashion MNIST dataset:](#train-on-fashion-mnist-dataset)
    - [Train on SST dataset:](#train-on-sst-dataset)
- [Results](#results)

# UvA_FACT_2021
![Z_0 Zoomed Sweep for MNIST 1/4/9](./figures/MNIST_149/pretrained/z_0_zoomed.png?raw=true)

Assignment for Fairness, Accountability, Confidentiality and Transparency in AI-University of Amsterdam.
The goal of this project is to assess the reproducibility of NeurIPS 2020 paper '[Generative Causal Explanations of Black-box Classifiers](https://arxiv.org/abs/2006.13913)', [[Github Repo](https://github.com/siplab-gt/generative-causal-explanations)]. Our submission to the ML Reproducibility Challenge 2020, detailing the reproducibility and extensibility of the proposed framework, may be found [here](./FACT_Report.pdf).

All results from the original paper have been implemented here, and extended to all discussed datasets (see details for generating figures below). An unsuccesful extension of GCEs to the Stanford sentiment treebank reviews is also included. The codebase makes extensive use of PyTorch, PyTorch Text and PyTorch Lightning.

## Report Conclusion
> While some issues and discrepancies were encountered while re-implementing, ultimately we conclude that the original paper combined with the official repository are enough to validate the claims of O'Shaugnessy et al. (2020). Results were comparable, and indeed led to high-quality explanations. However, while the central idea is elegant and is now proven to work, we bring into doubt the extensibility of their approach.  Due to the computational expense required, it is likely that the GCE models introduced will only function, in their current implementation, for small datasets and simple classifiers. Finally,this project confirms just how difficult it is to make implementations of AI transparent and reproducible.

# Structure
```
UvA_FACT_2021
├── checkpoints
│   └── model training output, including lightning/tensorboard logs and pretrained weights
├── datasets
│   ├── raw data sets in named directories
│   ├── mnist.py
│   │       file that contains code process MNIST, limiting it to specified classes only
│   ├── fashion_mnist.py
│   │       file that contains code process FMNIST, limiting it to specified classes only
│   ├── sst.py
│   └──     file that processes SST. Note, this makes use of TorchText classes that will get deprecated
├── figures
│   │     all figures produced during training, or otherwise relevant to the project Maybe should be placed in root directory
│   └──   PyTorch (Lightning) code for creating all training all models described
├── models
│   ├── mnist_cnn.py
│   │       models for classifying the limitied MNIST datasets
│   ├── mnist_cvae.py
│   │     cvae for explaining the limitied MNIST classifier
│   ├── sst_lstm_cnn.py
│   │       classifier architecture for SST reviews (BiLSTM fed into CNN)
│   ├── lm_vae.py
│   │       VAE using LM architectures to produce SST reviews
│   ├── lm_gce.py
│   └──     GCE for language explanation
├── pretrained_models
│   └── pretrained models (within upload limits) are stored here
├── utils
│   ├── cvae_latent_visualization.py
│   │       module for generating latent variable sweeps for a cvae
│   ├── information_flow.py
│   │       approximate mutual information computation using OShaugnessy et al.'s source code [UPDATED]
│   ├── reproducibility.py
│   │       module with some methods for reproducibility, including universal seed setting, model loading, etc.
│   ├── vae_loss.py
│   │       module for computing variational loss (ELBO, KLD, BPD)
│   ├── lagging_encoder.py
│   │       module for ensuring lagging encoder implementation functions
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
├── sst_classifier_train.py
│   └── file for training the SST classifier
├── lm_vae.py
│   └── file for training the language VAE
├── lm_gce.py
│   └── file for training the language GCE (needs pretrained VAE and classifier)
├── FACT_report.pdf
│   └── accompanying PDF report
└── results.ipynb
    └── notebook to easily reproduce obtained results
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
We provide a notebook (results.ipynb) to easily reproduce obtained results (this should be sufficient), and one single python file for generating all the results (including figure 5ab in the original paper for information flow and comparison of accuracies). Running the following command will save all plots to the figures/ folder:
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
### Train on SST dataset:
Since SST reviews are stored as trees, it is necessary to add NLTK to the environment. This should be done before running the remaining code files
```
pip install nltk
```

1. For training classifier
```
run python sst_classifier_train.py --max_epochs 15
```

2. For training text-VAE
```
srun python lm_vae_train.py --max_epochs 100 --inner_iter 250 --max_aggressive_epochs 25 --min_scheduler_epoch 15 --warm_up 10 --sample_every 5
```

3. For fine-tuning GCE (NOTE, requires pre-trained classifier and text-VAE)
```
run python lm_gce_train.py --max_epochs 15 --batch_size 64 --Nalpha 32 --Nbeta 8 --lr 1e-3 --lamb 1e-3
```

# Results
All the results are saved in ./figures directory.

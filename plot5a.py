import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.utils.data as data

from models.mnist_cnn import MNIST_CNN
from utils.reproducibility import set_seed, set_deteministic
from datasets.mnist import MNIST_limited

from mnist_cvae_train import GenerateCallback
from models.cvae import MNIST_CVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt

import numpy as np
import torch.nn.functional as F

def train(args):
    """
    Inputs:
        args - Namespace object from the argparser
    """

    M = len(args.classes)

    train_set, valid_set = MNIST_limited(train=True, classes=args.classes)
    test_set = MNIST_limited(train=False, classes=args.classes)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, pin_memory=True, num_workers=0)

    # load classifier
    classifier = MNIST_CNN(model_param_set=args.clf_param_set, M=M,
                        lr=args.lr, momentum=args.momentum)

    classifier_path = './pretrained_models/mnist_cnn/'
    checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
    classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

    # load GCE
    gce_path = './pretrained_models/mnist_cvae/'
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)

    # plot information_flow
    z_dim = args.K + args.L
    info_flow = gce.information_flow_single(range(0,z_dim))
    print("hi there!", info_flow)
    cols = {'golden_poppy' : [1.000,0.761,0.039],
        'bright_navy_blue' : [0.047,0.482,0.863],
        'rosso_corsa' : [0.816,0.000,0.000]}
    x_labels = ('$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$')
    fig, ax = plt.subplots()
    ax.bar(range(z_dim), info_flow, color=[
        cols['rosso_corsa'], cols['rosso_corsa'], cols['bright_navy_blue'],
        cols['bright_navy_blue']])
    plt.xticks(range(z_dim), x_labels)
    ax.yaxis.grid(linewidth='0.3')
    plt.ylabel('Information flow to $\\widehat{Y}$')
    plt.title('Information flow of individual causal factors')
    plt.savefig('./figures/fig5a.svg')
    plt.savefig('./figures/fig5a.pdf')
    print("done 5a")

    # --- load test data ---
    # train_set, valid_set = MNIST_limited(train=True, classes=args.classes)
    # test_set = MNIST_limited(train=False, classes=args.classes)
    #
    # train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    #                                drop_last=True, pin_memory=True, num_workers=0)
    # valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
    #                                drop_last=True, pin_memory=True, num_workers=0)
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
    #                               drop_last=True, pin_memory=True, num_workers=0)
    #
    # data_classes = np.array([1,4,9])
    # y_dim = data_classes.shape[0]
    # ylabels = range(0,y_dim)

    # dataloader_iterator = iter(valid_loader)
    # vaX, vaY = next(dataloader_iterator)
    # print(vaX.shape, vaY.shape, z.shape)
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=0)
    X = train_set.data
    Y = train_set.targets
    vaX = valid_set.data
    vaY = valid_set.targets

    # x1 = np.zeros((10, 10))
    # x2 = X[None, :, :]


    # print(vaX.shape)
    # X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
    # vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
    # print(X.shape)
    ntrain, nrow, ncol = X.shape
    x_dim = nrow*ncol

    # --- compute classifier accuracy after 'removing' latent factors ---
    classifier_accuracy_original = np.zeros(z_dim)
    Yhat = np.zeros((len(vaX)))
    Yhat_reencoded = np.zeros((len(vaX)))
    Yhat_aspectremoved = np.zeros((z_dim, len(vaX)))
    # print("hello", len(vaX))
    for i_samp in range(len(vaX)):
        if (i_samp % 1000) == 0:
            print(i_samp)
        dataloader_iterator = iter(valid_loader)
        vaX1, vaY1 = next(dataloader_iterator)
        x = torch.from_numpy(np.asarray(vaX1)).float()
        # x = torch.from_numpy(vaX[i_samp:i_samp+1,:,:,:]).permute(0,3,1,2).float().to(device)
        # print("test here: ", x.type())
<<<<<<< HEAD

=======
        
>>>>>>> 449b03c8dcbc108a5bf235c1f799a2c10eb0d508
        Yhat[i_samp] = np.argmax(F.softmax(classifier(x), dim=1).cpu().detach().numpy())
        z = gce.encoder(x.to(device))[0]
        xhat = gce.decoder(z)
        xhat = torch.sigmoid(xhat)
        Yhat_reencoded[i_samp] = np.argmax(F.softmax(classifier(xhat.cpu()), dim=1).cpu().detach().numpy())
        for i_latent in range(z_dim):
            z = gce.encoder(x.to(device))[0]
            z[0,i_latent] = torch.randn((1))
            xhat = gce.decoder(z)
            xhat = torch.sigmoid(xhat)
            Yhat_aspectremoved[i_latent,i_samp] = np.argmax(F.softmax(classifier(xhat.cpu()), dim=1).cpu().detach().numpy())
    vaY = np.asarray(vaY)
    Yhat = np.asarray(Yhat)
    Yhat_reencoded = np.asarray(Yhat_reencoded)

    classifier_accuracy = np.mean(vaY == Yhat)
    classifier_accuracy_reencoded = np.mean(vaY == Yhat_reencoded)
    classifier_accuracy_aspectremoved = np.zeros((z_dim))
    for i in range(z_dim):
        classifier_accuracy_aspectremoved[i] = np.mean(vaY == Yhat_aspectremoved[i,:])

    print(classifier_accuracy, classifier_accuracy_reencoded, classifier_accuracy_aspectremoved)

    # --- plot classifier accuracy ---
    cols = {'black' : [0.000, 0.000, 0.000],
            'golden_poppy' : [1.000,0.761,0.039],
            'bright_navy_blue' : [0.047,0.482,0.863],
            'rosso_corsa' : [0.816,0.000,0.000]}
    x_labels = ('orig','reenc','$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$')
    fig, ax = plt.subplots()
    ax.yaxis.grid(linewidth='0.3')
    ax.bar(range(z_dim+2), np.concatenate(([classifier_accuracy],
                                           [classifier_accuracy_reencoded],
                                           classifier_accuracy_aspectremoved)),
           color=[cols['black'], cols['black'], cols['rosso_corsa'],
                  cols['rosso_corsa'], cols['bright_navy_blue']])
    plt.xticks(range(z_dim+2), x_labels)
    plt.ylim((0.2,1.0))
    plt.yticks((0.2,0.4,0.6))#,('0.5','','0.75','','1.0'))
    plt.ylabel('Classifier accuracy')
    plt.title('Classifier accuracy after removing aspect')
    plt.savefig('./figures/fig5b.svg')
    plt.savefig('./figures/fig5b.pdf')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--clf_param_set', default='OShaugnessy',
                        type=str, help='The black-box classifier we wish to explain.')
    parser.add_argument('--classes', default=[1, 4, 9],
                        type=int, nargs='+',
                        help='The classes permittible for classification')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--max_epochs', default=50, type=int,
                        help='Max number of training epochs')

    # Other hyperparameters

    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--progress_bar', default=True, action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--log_dir', default='mnist_cnn', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--add_classes_to_cpt_path', default=True,
                        help='Whether to add the classes to cpt directory.')


    # Debug parameters
    parser.add_argument('--debug_version', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--fast_dev_run', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    # param for cvae
    parser.add_argument('--K', default=2, type=int,
                       help='Dimensionality of causal latent space')
    parser.add_argument('--L', default=2, type=int,
                       help='Dimensionality of non-causal latent space')
    parser.add_argument('--M', default=3, type=int,
                       help='Dimensionality of classifier output')

    args = parser.parse_args()

    train(args)

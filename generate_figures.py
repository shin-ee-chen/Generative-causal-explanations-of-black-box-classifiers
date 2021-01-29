from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.rcParams["text.usetex"]

from datasets.mnist import *
from datasets.fashion_mnist import Fashion_MNIST_limited
from mnist_cvae_train import GenerateCallback
from models.cvae import MNIST_CVAE
from models.mnist_cnn import MNIST_CNN
from utils.reproducibility import load_latest, set_seed
from utils.vae_loss import *

FIGURE_PATH = './figures'
PAGE_WIDTH = 10

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def generate_figures(implementation, seed=42, rows=8, cols=7, shuffle=True,
                     single_sweep_ok=True, class_sweep_ok=True, information_flows_ok=True,
                     ablation_accuracy_ok=True, zoomed_ok=True):

    colors = ['#FFC209', '#0B7ADC', '#8FC839', 'tab:red']

    set_seed(seed)

    GCEs = {'MNIST_38': MNIST_CVAE, 'MNIST_149': MNIST_CVAE}
    CLFs = {'MNIST_38': MNIST_CNN, 'MNIST_149': MNIST_CNN}
    if implementation.upper() not in GCEs.keys():
        print('Not implemented. Please choose one from', GCEs.keys())
        # return None

    [dataset_name, classes_str] = implementation.rsplit('_')
    dataset_name = dataset_name.upper()
    classes = [int(c) for c in classes_str]

    if dataset_name == 'MNIST':
        dataset = MNIST_limited(train=False, classes=classes)

    elif dataset_name == 'FMNIST':
        # TODO: add FMNIST
        dataset = MNIST_limited(train=False, classes=classes)

    loader = data.DataLoader(dataset, batch_size=64, shuffle=shuffle)
    loader = iter(loader)

    x, y = next(loader)

    gce_name = dataset_name.lower() + '_' + 'gce' + '_' + classes_str
    gce = load_latest(trainer=GCEs[implementation.upper()],
                    save_name=gce_name,
                    map_location=torch.device('cpu'),
                    inference=True)

    clf_name = dataset_name.lower() + '_' + 'cnn' + '_' + classes_str
    clf = load_latest(trainer=CLFs[implementation.upper()],
                    save_name=clf_name,
                    map_location=torch.device('cpu'),
                    inference=True)

    z_dim = min(gce.K + gce.L, rows)

    ########################################################################
    ### Plot sweep over single latent for ROWS images ######################
    ########################################################################

    def latent_sweep_single_factor(zi, zoomed=False):

        if zoomed:
            fig, axes = plt.subplots(1, 1, figsize=(PAGE_WIDTH, PAGE_WIDTH / 4))
            n_cols = 39
            vals = np.linspace(-2, 2, num=n_cols)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(PAGE_WIDTH / 4, PAGE_WIDTH / 3))
            n_cols = cols
            vals = np.linspace(-3, 3, num=n_cols)

        imgs, label_idx = [], []

        stats = gce.encoder(x[:rows])
        z = sample_reparameterize(*stats)
        for val in vals:
            z_ = z.clone()
            z_[:, zi] += val
            xhat = torch.sigmoid(gce.decoder(z_))
            yhat = torch.argmax(clf(xhat), dim=-1)

            imgs.append(xhat.permute(0, 2, 3, 1))
            label_idx.append(yhat.numpy())

        imgs, label_idx = np.stack(imgs, axis=1), np.stack(label_idx, axis=1)

        label = 'alpha_{:d}'.format(zi + 1) if zi + 1 <= gce.K else \
            'beta_{:d}'.format(zi + 1 - gce.K)

        grid = ImageGrid(fig, 111, nrows_ncols=(rows, n_cols), aspect=True)
        grid.set_axes_pad([0.05, 0.05])

        i = 0
        for n_row, row in enumerate(imgs):
            for n_col, img in enumerate(row):

                grid[i].imshow(img.squeeze(), cmap='binary', vmin=0, vmax=1)

                grid[i].tick_params(axis='both', which='both',
                            bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)

                for axis in ['top', 'bottom', 'left', 'right']:
                    grid[i].spines[axis].set_linewidth(2.5)
                    grid[i].spines[axis].set_edgecolor(colors[label_idx[n_row, n_col]])

                i += 1

        for axis in ['top', 'left', 'right']:
            axes.spines[axis].set_visible(False)

        axes.tick_params(axis='both', which='both',
                        bottom=True, top=False, left=False, right=False,
                        labelbottom=True, labelleft=False)

        vals_step = np.max(vals) / (n_cols - 1)
        axes.set_xlim(np.min(vals) - vals_step, np.max(vals) + vals_step)

        major_vals = np.linspace(np.min(vals), np.max(vals), int(2 * np.max(vals) + 1))
        axes.set_xticks(vals, minor=True)
        axes.set_xticks(major_vals)
        axes.set_xticklabels(['{:+.1f}'.format(x) for x in major_vals], fontsize=9)

        fig.suptitle('Sweep over $\\{:s}$'.format(label), y=0.9, fontsize=11)

        fig_dir = os.path.join(FIGURE_PATH, implementation.upper(), 'pretrained')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(fname=os.path.join(fig_dir, 'z_{:d}_zoomed.pdf'.format(zi) if zoomed else 'z_{:d}.pdf'.format(zi)))

        plt.close()

    ########################################################################
    ### Plot sweep over all latents for single image #######################
    ########################################################################

    def class_latent_sweep(class_int):

        internal_class = int(np.where(np.array(classes) == class_int)[0])

        x_select = torch.stack([(x[y == i])[0] for i in range(len(classes))])

        fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH / 4, PAGE_WIDTH / 3))

        axes[0].imshow(x_select[internal_class].permute(1, 2, 0).squeeze(), cmap='binary')

        vals = np.linspace(-3, 3, num=cols)

        stats = gce.encoder(x_select[internal_class].unsqueeze(0))
        z = sample_reparameterize(*stats)
        imgs, label_idx = [], []
        for zi in range(z_dim):
            imgs_inner, label_idx_inner = [], []
            for val in vals:
                z_ = z.clone()
                z_[:, zi] += val
                xhat = torch.sigmoid(gce.decoder(z_))
                yhat = torch.argmax(clf(xhat), dim=-1)

                imgs_inner.append(xhat.permute(0, 2, 3, 1)[0].numpy())
                label_idx_inner.append(yhat[0].numpy())

            imgs.append(imgs_inner)
            label_idx.append(label_idx_inner)
        imgs = np.stack(imgs, axis=0)
        label_idx = np.stack(label_idx, axis=0)

        grid = ImageGrid(fig, 212, nrows_ncols=(z_dim, cols), aspect=False)
        grid.set_axes_pad([0.05, 0.05])

        i = 0
        for n_row, row in enumerate(imgs):
            for n_col, img in enumerate(row):

                grid[i].imshow(img.squeeze(), cmap='binary', vmin=0, vmax=1)

                for axis in ['top', 'bottom', 'left', 'right']:
                    grid[i].spines[axis].set_linewidth(2.5)
                    grid[i].spines[axis].set_edgecolor(colors[label_idx[n_row, n_col]])

                if n_col == 0:
                    label = '$\\alpha_{:d}$'.format(n_row + 1) if n_row + 1 <= gce.K else \
                        '$\\beta_{:d}$'.format(n_row + 1 - gce.K)
                    grid[i].set_ylabel(label)

                grid[i].tick_params(axis='both', which='both',
                                    bottom=False, top=False, left=False, right=False,
                                    labelbottom=False, labelleft=False)

                i += 1

        for axis in ['top', 'left', 'right']:
            axes[1].spines[axis].set_visible(False)

        axes[0].tick_params(axis='both', which='both',
                            bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)
        axes[1].tick_params(axis='both', which='both',
                            bottom=True, top=False, left=False, right=False,
                            labelbottom=True, labelleft=False)

        vals_step = np.max(vals) / (cols - 1)
        axes[1].set_xlim(np.min(vals) - vals_step, np.max(vals) + vals_step)

        major_vals = np.linspace(np.min(vals), np.max(vals), int(2 * np.max(vals) + 1))
        axes[1].set_xticks(vals, minor=True)
        axes[1].set_xticks(major_vals)
        axes[1].set_xticklabels(['{:+.1f}'.format(x) for x in major_vals], fontsize=9)
        axes[1].spines['bottom'].set_position(('outward', 10))

        fig.suptitle('Latent sweeps for class {:d}'.format(class_int), y=0.95, fontsize=11)

        fig_dir = os.path.join(FIGURE_PATH, implementation.upper(), 'pretrained')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(fname=os.path.join(fig_dir, 'class_{:d}_sweep.pdf'.format(class_int)))

        plt.close()

    ########################################################################
    ### Ablation study plots ###############################################
    ########################################################################

    def information_flows():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and implementation == 'fmnist_034':  ## because the model is trained on gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # information flow ablation study for fmnist data (figure 5a in the paper)
            M = 3
            K = 2
            L = 4

            # load GCE
            gce_path = './pretrained_models/fmnist_gce_034/'
            gce = torch.load(os.path.join(gce_path,'gce_model.pt'), map_location=device)
            print("gce pretrained model loaded!")

            # plot information_flow
            z_dim = K + L
            info_flow = gce.information_flow_single(range(0,z_dim))

            # we use author's code for making the exact same plot
            cols = {'golden_poppy' : [1.000,0.761,0.039],
                    'bright_navy_blue' : [0.047,0.482,0.863],
                    'rosso_corsa' : [0.816,0.000,0.000]}
            x_labels = ('$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$', '$\\beta_3$', '$\\beta_4$')
            fig, ax = plt.subplots()
            ax.bar(range(z_dim), info_flow, color=[
                cols['rosso_corsa'], cols['rosso_corsa'], cols['bright_navy_blue'],
                cols['bright_navy_blue'], cols['bright_navy_blue'], cols['bright_navy_blue']])
            plt.xticks(range(z_dim), x_labels)
            ax.yaxis.grid(linewidth='0.3')
            plt.ylabel('Information flow to $\\widehat{Y}$')
            plt.title('Information flow of individual causal factors')
            fig_dir = os.path.join(FIGURE_PATH, implementation.upper(), 'pretrained')
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(fname=os.path.join(fig_dir, 'InformationFlow.pdf'))
            plt.close()

        else:
            print("You need cuda to make this plot because all our models are trained on gpu.")

    def ablation_accuracy():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and implementation == 'fmnist_034':  ## because the model is trained on gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # accuracy comparison ablation study for fmnist data (figure 5b in the paper)
            # load GCE
            gce_path = './pretrained_models/fmnist_gce_034/'
            gce = torch.load(os.path.join(gce_path,'gce_model.pt'), map_location=device)
            print("gce pretrained model loaded!")
            # load classifier
            classifier = MNIST_CNN(model_param_set='OShaugnessy', M=M, lr=5e-4, momentum=0.9)
            classifier_path = './pretrained_models/fmnist_cnn_034/'
            checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
            classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])
            print("classifier pretrained model loaded!")

            # --- load test data ---
            train_set, valid_set = Fashion_MNIST_limited(train=True, classes=[0,3,4])
            valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False,
                                        drop_last=True, pin_memory=True, num_workers=0)

            X = train_set.data
            Y = train_set.targets
            vaX = valid_set.dataset.data
            vaY = valid_set.dataset.targets

            ntrain, nrow, ncol = X.shape
            x_dim = nrow*ncol

            # compute classifier accuracy after removing latent factors (minor changes compared to author's code)
            classifier_accuracy_original = np.zeros(z_dim)
            Yhat = np.zeros((len(vaX)))
            Yhat_reencoded = np.zeros((len(vaX)))
            Yhat_aspectremoved = np.zeros((z_dim, len(vaX)))

            for i_samp in range(len(vaX)):
                if (i_samp % 1000) == 0:
                    print(i_samp)
                dataloader_iterator = iter(valid_loader)
                vaX1, vaY1 = next(dataloader_iterator)
                x = torch.from_numpy(np.asarray(vaX[None, i_samp:i_samp+1,:,:])).float().to(device)

                Yhat[i_samp] = np.argmax(F.softmax(classifier(x.cpu()), dim=1).cpu().detach().numpy())
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

            # plot classifier accuracy
            # we use author's code for making the exact same plot
            cols = {'black' : [0.000, 0.000, 0.000],
                    'golden_poppy' : [1.000,0.761,0.039],
                    'bright_navy_blue' : [0.047,0.482,0.863],
                    'rosso_corsa' : [0.816,0.000,0.000]}
            x_labels = ('orig','reenc','$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$',
                        '$\\beta_3$', '$\\beta_4$')
            fig, ax = plt.subplots()
            ax.yaxis.grid(linewidth='0.3')
            ax.bar(range(z_dim+2), np.concatenate(([classifier_accuracy],
                                                [classifier_accuracy_reencoded],
                                                classifier_accuracy_aspectremoved)),
                color=[cols['black'], cols['black'], cols['rosso_corsa'],
                        cols['rosso_corsa'], cols['bright_navy_blue'],
                        cols['bright_navy_blue'], cols['bright_navy_blue'],
                        cols['bright_navy_blue']])
            plt.xticks(range(z_dim+2), x_labels)
            plt.ylim((0.2,1.0))
            plt.yticks((0.2,0.4,0.6,0.8,1.0))#,('0.5','','0.75','','1.0'))
            plt.ylabel('Classifier accuracy')
            plt.title('Classifier accuracy after removing aspect')
            fig_dir = os.path.join(FIGURE_PATH, implementation.upper(), 'pretrained')
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(fname=os.path.join(fig_dir, 'AblationAccuracy.pdf'))
            plt.close()
        else:
            print("You need cuda to make this plot because all our models are trained on gpu.")

    ########################################################################
    ### Internal function calls ############################################
    ########################################################################

    if single_sweep_ok:
        for zi in range(z_dim):
            latent_sweep_single_factor(zi)

    if class_sweep_ok:
        for c in classes:
            class_latent_sweep(c)

    if information_flows_ok:
        information_flows()

    if ablation_accuracy_ok:
        ablation_accuracy()

    if zoomed_ok:
        for zi in range(gce.K):
            latent_sweep_single_factor(zi=zi, zoomed=True)


generate_figures(implementation='mnist_149')
generate_figures(implementation='mnist_38')

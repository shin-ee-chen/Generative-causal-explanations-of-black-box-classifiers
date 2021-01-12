import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.mnist_cnn import MNIST_CNN
from utils.reproducibility import load_latest
from utils.vae_loss import sample_reparameterize, ELBO, ELBO_to_BPD
from utils.information_flow import CVAE_to_params, joint_uncond


class CNN_Encoder(nn.Module):
    def __init__(self, img_channels: int = 1, num_filters: int = 64,
                 latent_dim: int = 8):
        """
        Encoder with a CNN network. Defaults to the OShaugnessy architecture in Table 2, Appendix E

        Inputs:
            img_channels - Number of input channels of the image.
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            latent_dim - The number of latents to encode into. Note, this is K+L.
        """
        super().__init__()

        c_hid = num_filters
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, num_filters, kernel_size=4,
                      padding=1, stride=2),  # 28x28 => 14x14
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4,
                      padding=1, stride=2),  # 14x14 => 7x7
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4,
                      padding=0, stride=1),  # 7x7 => 4x4
            nn.ReLU(),
            nn.Flatten(),  # Image grid to single feature vector
        )

        self.mean = nn.Linear(4 * 4 * num_filters, latent_dim)
        self.log_std = nn.Linear(4 * 4 * num_filters, latent_dim)

        for layer in [self.net.children()] + [self.mean] + [self.log_std]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, imgs):
        """
        Inputs:
            imgs - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        hidden = self.net(imgs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)

        return mean, log_std

class CNN_Decoder(nn.Module):
    def __init__(self, img_channels: int = 1, num_filters: int = 64,
                 latent_dim: int = 8):
        """
        Decoder with a CNN network. Defaults to the OShaugnessy architecture in Table 2, Appendix E

        Inputs:
            img_channels - Number of input channels of the image.
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            latent_dim - The number of latents to encode into. Note, this is K+L.
        """
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.ReLU()
        )

        self.convnet = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4,
                               padding=1, stride=1),  # 7x7 => 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4,
                               padding=2, stride=2),  # 8x8 => 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, img_channels, kernel_size=4,
                               padding=1, stride=2),  # 14x14 => 28x28
        )

        for layer in [self.linear.children()] + [self.convnet.children()]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z. Shape: [B,num_input_channels,28,28]
        """

        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x = self.convnet(x)

        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        """
        return next(self.parameters()).device

class MNIST_CVAE(pl.LightningModule):

    def __init__(self, classes, num_filters, K, L, M, lamb, lr, Nalpha, Nbeta, betas, classifier_path=None):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            classifier_name - String denoting what classifier black-box to use.
            num_filters - Number of channels to use in a CNN encoder/decoder
            K - Number of causal latent variables
            L - Number of non-causal latent variables
            M - Number of classifier classes
            lamb - The lambda coefficient controlling how much influence the VAE loss has
            lr - Learning rate to use for the optimizer
            Nalpha - Number of samples used to approximate the information flow loss
            Nbeta - Same as above, but then for the non-causal latent variables
        """
        super().__init__()
        self.save_hyperparameters()

        self.K = K
        self.L = L
        self.M = M
        self.lamb = lamb
    
        self.latent_dim = self.K + self.L

        self.Nalpha = Nalpha
        self.Nbeta = Nbeta
        self.lr = lr
        self.betas = tuple(betas)
                
        self.encoder = CNN_Encoder(img_channels=1, num_filters=num_filters, latent_dim=K + L)
        self.decoder = CNN_Decoder(img_channels=1, num_filters=num_filters, latent_dim=K + L)
        
        self.classes_str = ''.join(str(x) for x in sorted(classes))
        
        if classifier_path == None:
            self.classifier = load_latest(MNIST_CNN, 'mnist_cnn_'+self.classes_str,
                                        inference=True, map_location=self.device)
        else:
            self.classifier = load_latest(MNIST_CNN, classifier_path,
                                         inference=True, map_location=self.device)
            

    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - the average reconstruction loss of the batch.
            L_reg - the average Kullback-Leibler divergence.
            C = the causal loss term, i.e. mutual information.
            bpd - the average bits per dimension metric of the batch.
        """

        imgs = imgs.to(self.device)

        mean, log_std = self.encoder(imgs)
        z = sample_reparameterize(mean, log_std)
        x_hat = torch.sigmoid(self.decoder(z))

        elbo, L_rec, L_reg = ELBO(imgs, x_hat, mean, log_std)

        C = self.information_flow()

        bpd = ELBO_to_BPD(elbo, imgs.size())

        return torch.mean(L_rec), torch.mean(L_reg), C, torch.mean(bpd)

    def information_flow(self):
        """
        Computes approximate mutual information between the classifier and the designated causal variables.

        Returns:
            C : the causal loss term, otherwise, mutual information: I(alpha; Y)
        """

        C, debug = joint_uncond(*CVAE_to_params(self))

        return C

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     betas= self.betas
                                     )
        return optimizer

    def training_step(self, batch, batch_idx):

        L_rec, L_reg, C, bpd = self.forward(batch[0])

        self.log("Train Reconstruction Loss", L_rec,
                 on_step=True, on_epoch=False)
        self.log("Train Regularization Loss", L_reg,
                 on_step=True, on_epoch=False)
        self.log("Train Information Flow", C, on_step=True, on_epoch=False)

        self.log("Train ELBO", L_rec + L_reg, on_step=True, on_epoch=False)
        self.log("Train Causal Loss", C + self.lamb * (L_rec + L_reg), on_step=True, on_epoch=False)
        self.log("Train BPD", bpd, on_step=True, on_epoch=False)

        return C + self.lamb * (L_rec + L_reg)  # bpd

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, C, bpd = self.forward(batch[0])
        self.log("Valid Reconstruction Loss", L_rec,
                 on_step=False, on_epoch=True)
        self.log("Valid Regularization Loss", L_reg,
                 on_step=False, on_epoch=True)
        self.log("Valid Information Flow", C, on_step=False, on_epoch=True)

        self.log("Valid ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("Valid Causal Loss", C + self.lamb * (L_rec + L_reg), on_step=False, on_epoch=True)
        self.log("Valid BPD", bpd, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, C, bpd = self.forward(batch[0])
        self.log("Test Information Flow", C, on_step=False, on_epoch=True)
        self.log("Test ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("Test BPD", bpd, on_step=False, on_epoch=True)
        self.log("Test Causal Loss", C + self.lamb * (L_rec + L_reg), on_step=False, on_epoch=True)

    @torch.no_grad()
    def sample(self, batch_size: int):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s. Shape: [B,C,H,W]
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples".
                     Shape: [B,C,H,W]
        """

        z = torch.normal(0, 1, size=(batch_size, self.z_dim),
                         device=self.device)
        x_mean = torch.sigmoid(self.decoder(z)).detach()
        x_samples = torch.round(x_mean)

        return x_samples, x_mean

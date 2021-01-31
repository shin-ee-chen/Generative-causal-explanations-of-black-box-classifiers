import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.sst_bilstm_cnn import sst_bilstm_cnn
from models.lm_vae import lm_VAE
from utils.information_flow import joint_uncond
from utils.reproducibility import load_latest

class lm_gce(pl.LightningModule):

    def __init__(self, clf_path, vae_path, K, lamb, Nalpha, Nbeta, lr, betas):

        super().__init__()
        self.save_hyperparameters()

        if clf_path == '':
            clf = load_latest(sst_bilstm_cnn, 'sst_lstm_cnn_2', inference=True)
        else:
            clf = load_latest(sst_bilstm_cnn, clf_path, inference=True)

        if vae_path == '':
            vae = load_latest(lm_VAE, 'text_vae')
        else:
            vae = load_latest(lm_VAE, vae_path)

        self.classifier = clf
        self.vae = vae

        self.K = K
        self.lamb = lamb
        self.Nalpha = Nalpha
        self.Nbeta = Nbeta
        self.betas = betas
        self.lr = lr


    def forward(self, batch):

        L_rec, L_reg = self.vae.forward(batch)

        C = self.information_flow()

        return torch.mean(L_rec), torch.mean(L_reg), C

    def information_flow(self):
        """
        Computes approximate mutual information between the classifier and the designated causal variables.

        Returns:
            C : the causal loss term, otherwise, mutual information: I(alpha; Y)
        """

        def _lm_gce_to_params(self):
            params = dict()

            params["z_dim"] = self.vae.latent_dims
            params["alpha_dim"] = self.vae.latent_dims
            params["Nalpha"] = self.Nalpha
            params["Nbeta"] = self.Nbeta
            params["decoder_net"] = 'VAE'
            params["M"] = self.classifier.M
            params["K"] = self.K
            params["L"] = self.vae.latent_dims - self.K
            params["break_up_ce"] = True

            decoder = self.vae.decoder
            classifier = self.classifier

            return params, decoder, classifier, self.device


        C, debug = joint_uncond(*_lm_gce_to_params(self), argmax=True)

        return C

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.vae.parameters(),
                                     lr=self.lr,
                                     betas=self.betas
                                    )
        return optimizer

    def training_step(self, batch, batch_idx):

        L_rec, L_reg, C = self.forward(batch)
        causal_loss = C + self.lamb * (L_rec + L_reg)

        self.log("Train Reconstruction Loss", L_rec)
        self.log("Train Regularization Loss", L_reg)
        self.log("Train Information Flow", C)

        self.log("Train ELBO", L_rec + L_reg)
        self.log("Train Causal Loss", causal_loss)

        return causal_loss

    def validation_step(self, batch, batch_idx):

        L_rec, L_reg, C = self.forward(batch)
        causal_loss = C + self.lamb * (L_rec + L_reg)

        self.log("Valid Reconstruction Loss", L_rec, on_epoch=True)
        self.log("Valid Regularization Loss", L_reg, on_epoch=True)
        self.log("Valid Information Flow", C, on_epoch=True)

        self.log("Valid ELBO", L_rec + L_reg, on_epoch=True)
        self.log("Valid Causal Loss", causal_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):

        L_rec, L_reg, C = self.forward(batch)
        causal_loss = C + self.lamb * (L_rec + L_reg)

        self.log("Test Reconstruction Loss", L_rec, on_epoch=True)
        self.log("Test Regularization Loss", L_reg, on_epoch=True)
        self.log("Test Information Flow", C, on_epoch=True)

        self.log("Test ELBO", L_rec + L_reg, on_epoch=True)
        self.log("Test Causal Loss", causal_loss, on_epoch=True)

    @torch.no_grad()
    def sample(self, batch_size: int):

        z = torch.normal(0, 1, size=(batch_size, self.z_dim),
                         device=self.device)
        x_mean = torch.sigmoid(self.decoder(z)).detach()
        x_samples = torch.round(x_mean)

        return x_samples, x_mean

    @torch.no_grad()
    def decode(self, text, decoding_strategy=None, beam_length=5):

        text_sample = self.vae.decode(text, decoding_strategy=decoding_strategy, beam_length=beam_length)

        return text_sample

    @torch.no_grad()
    def latent_sweep(self, text, zi, num=7, decoding_strategy=None, beam_length=5, tau=0.5):

        sweep_text = self.vae.latent_sweep(text, zi, num=num, decoding_strategy=decoding_strategy,
                                           beam_length=beam_length, tau=tau)

        return sweep_text

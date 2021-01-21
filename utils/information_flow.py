import numpy as np
import torch
import torch.nn.functional as F

def CVAE_to_params(CVAE):

    params = dict()

    params["z_dim"] = CVAE.latent_dim
    params["alpha_dim"] = CVAE.K
    params["Nalpha"] = CVAE.Nalpha
    params["Nbeta"] = CVAE.Nbeta
    params["decoder_net"] = 'VAE'
    params["M"] = CVAE.M
    params["K"] = CVAE.K
    params["L"] = CVAE.L
    params["break_up_ce"] = True

    decoder = CVAE.decoder
    classifier = CVAE.classifier

    return params, decoder, classifier, CVAE.device

def joint_uncond(params, decoder, classifier, device):

    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M']).to(device)
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))

    for i in range(0, params['Nalpha']):
        alpha = np.random.randn(params['K'])
        zs = np.zeros((params['Nbeta'], params['z_dim']))
        for j in range(0, params['Nbeta']):
            beta = np.random.randn(params['L'])
            zs[j, :params['K']] = alpha
            zs[j, params['K']:] = beta

        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        xhat = torch.sigmoid(xhat)

        # yhat = classifier(xhat)[0]
        yhat = F.softmax(classifier(xhat), dim=1)
        p = 1./float(params['Nbeta']) * \
            torch.sum(yhat, 0)  # estimate of p(y|alpha)
        I = I + 1./float(params['Nalpha']) * \
            torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p  # accumulate estimate of p(y)

    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))

    negCausalEffect = -I
    info = {"xhat": xhat, "yhat": yhat}

    return negCausalEffect, info
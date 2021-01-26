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
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))

    for i in range(0, params['Nalpha']):
        alpha = np.random.randn(params['K'])
        for j in range(0, params['Nbeta']):
            beta = np.random.randn(params['L'])
            zs[i * params['Nbeta'] + j, :params['K']] = alpha
            zs[i * params['Nbeta'] + j, params['K']:] = beta

    # decode and classify samples
    xhat = decoder(torch.from_numpy(zs).float().to(device))
    xhat = torch.sigmoid(xhat)
    yhat = F.softmax(classifier(xhat), dim=1)
    
    yhats = torch.chunk(yhat, params['Nalpha'])
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M']).to(device)
    
    for i in range(0, params['Nalpha']):
        p = 1./float(params['Nbeta']) * torch.sum(yhats[i], 0)  # estimate of p(y|alpha)
        I += 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q += 1./float(params['Nalpha']) * p  # estimate of p(y)

    I -= torch.sum(torch.mul(q, torch.log(q+eps)))

    negCausalEffect = -I
    info = {"xhat": xhat, "yhat": yhat}

    return negCausalEffect, info


def joint_uncond_singledim(params, decoder, classifier, device, dims):

    eps = 1e-5
    I = 0.0

    q = torch.zeros(params['M']).to(device)
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))
    for i in range(0, params['Nalpha']):
        z_fix = np.random.randn(1)

        zs = np.zeros((params['Nbeta'],params['z_dim']))
        for j in range(0, params['Nbeta']):
            zs[j,:] = np.random.randn(params['K']+params['L'])
            zs[j,dims] = z_fix

        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        xhat = torch.sigmoid(xhat)
        yhat = F.softmax(classifier(xhat), dim=1)
        # yhat = classifier(xhat)[0]
        # yhat = yhat.cpu().detach().numpy().max()
        # print("here is my yhat!", yhat)

        p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
        # print("look here for debug1", float(params['Nbeta']), "end")
        I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
        # print("look here for debug2", I)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    negCausalEffect = -I
    info = {"xhat" : xhat, "yhat" : yhat}

    return negCausalEffect, info

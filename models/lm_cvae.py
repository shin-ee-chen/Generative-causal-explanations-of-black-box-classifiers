import math

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from utils.lagging_encoder import *
from utils.vae_loss import *

class LSTM_Encoder(nn.Module):

    def __init__(self, vocab, embedding_dims, hidden_dims, latent_dims):
        super(LSTM_Encoder, self).__init__()

        self.latent_dims = latent_dims
        self.vocab = vocab

        padding_idx=vocab['<pad>']
        self.embed = nn.Embedding(len(vocab), embedding_dims,
                                  padding_idx=padding_idx)

        self.lstm = nn.LSTM(input_size=embedding_dims,
                            hidden_size=hidden_dims,
                            num_layers=1,
                            dropout=0,
                            batch_first=False)

        self.mean = nn.Linear(in_features=hidden_dims, out_features=latent_dims)
        self.log_std = nn.Linear(in_features=hidden_dims, out_features=latent_dims)

        self.reset_parameters(std=0.01)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_parameters(self, std=0.01):
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        nn.init.uniform_(self.embed.weight, -std*10, std*10)

    def forward(self, input):

        # (batch_size, seq_len-1, args.ni)
        embedding = self.embed(input)

        _, (h_T, _) = self.lstm(embedding)

        mean = self.mean(h_T)
        log_std = self.log_std(h_T)

        return mean, log_std

    @torch.no_grad()
    def MutualInformation(self, x, stats=None, z_iters=1, debug=False):
        """
        Approximate the mutual information between x and z,
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z)).

        Adapted from https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/encoders/encoder.py

        Returns: Float
        """

        if stats == None:
            mean, logstd = self.forward(x)
        else:
            mean, logstd = stats

        logvar = 2 * logstd

        x_batch, z_dim = mean.shape[1], mean.shape[2]
        z_batch = x_batch * z_iters

        # E_{q(z|x)}log(q(z|x)) = -0.5*(K+L)*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * z_dim * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, z_dim]
        z_samples = []
        for i in range(z_iters):
            z_samples.append(sample_reparameterize(mean, logstd).permute(1, 0, 2))
        z_samples = torch.cat(z_samples, dim=0)
        if debug: print('[z_batch, 1, z_dim]', z_samples.shape)


        # [1, x_batch, z_dim]
        var = logvar.exp()
        if debug:
            print('[1, x_batch, z_dim]', mean.shape)

        # (z_batch, x_batch)
        log_density = -0.5 * (((z_samples - mean) ** 2) / var).sum(dim=-1) - \
            0.5 * (z_dim * math.log(2 * math.pi) + logvar.sum(-1))
        if debug:
            print('(z_batch, x_batch)', log_density.shape)

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)
        if debug: print('[z_batch]', log_qz.shape)

        return (neg_entropy - log_qz.mean(-1)).detach()

class LSTM_Decoder(nn.Module):

    def __init__(self, vocab, embedding_dims, hidden_dims, latent_dims):
        super(LSTM_Decoder, self).__init__()

        self.latent_dims = latent_dims
        self.vocab = vocab

        padding_idx=vocab['<pad>']
        self.embed = nn.Embedding(len(vocab), embedding_dims,
                                  padding_idx=padding_idx)

        self.dropout_in = nn.Dropout(p=0.5)
        self.linear_in = nn.Linear(self.latent_dims, hidden_dims)

        self.lstm=nn.LSTM(input_size=embedding_dims + self.latent_dims,
                          hidden_size=hidden_dims,
                          num_layers=1,
                          dropout=0,
                          batch_first=False)

        self.dropout_out = nn.Dropout(p=0.5)
        self.linear_out = nn.Linear(hidden_dims, len(vocab))

        self.reset_parameters(std=0.01)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_parameters(self, std=0.01):
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        nn.init.uniform_(self.embed.weight, -std*10, std*10)

    def forward(self, input, z, debug=False):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        n_sample, batch_size, z_dim = z.size()
        seq_len = input.size(0)

        # (seq_len, batch_size, embedding_dim)
        word_embed = self.embed(input)
        if debug: print('(seq_len, batch_size, embedding_dim)', word_embed.shape)

        word_embed = self.dropout_in(word_embed)

        z_ = z.expand(seq_len, batch_size, z_dim)

        # (seq_len, batch_size * n_sample, embedding_dim + z_dim)
        word_embed = torch.cat((word_embed, z_), -1)
        if debug: print('(seq_len, batch_size * n_sample, embedding_dim + z_dim)', word_embed.shape)

        z = z.view(batch_size * n_sample, z_dim)
        c_init = self.linear_in(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        output, _ = self.lstm(word_embed, (h_init, c_init))

        # (seq_len, batch_size * n_sample, vocab_size)
        output = self.dropout_out(output)
        if debug: print('(seq_len, batch_size * n_sample, vocab_size)', output.shape)
        output_logits = self.linear_out(output)

        return output_logits

    @torch.no_grad()
    def beam_search_decode(self, z, K=5, max_length=30):
        """beam search decoding, code is based on
        https://github.com/pcyin/pytorch_basic_nmt/blob/master/nmt.py
        the current implementation decodes sentence one by one, further batching would improve the speed
        Args:
            z: (batch_size, nz)
            K: the beam width
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size, nz = z.size()
        decoded_batch = [] #[[] for _ in range(batch_size)]

        # (1, batch_size, nz)
        c_init = self.linear_in(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            # Start with the start of the sentence token
            decoder_input = torch.tensor([[self.vocab["<s>"]]], dtype=torch.long, device=self.device)
            decoder_hidden = (h_init[:, idx, :].unsqueeze(0), c_init[:, idx, :].unsqueeze(0))

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0., 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < max_length:
                t += 1

                # (1, len(live))
                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=1)

                # (1, len(live), nh)
                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                # (1, len(live), ni) --> (1, len(live), ni+nz)
                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    1, len(live_hypotheses), nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                # (1, len(live), vocab_size)
                output_logits = self.linear_out(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses], dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(1, len(live_hypotheses), 1)

                # (len(live) * vocab_size)
                decoder_output = decoder_output.view(-1)

                # (K)
                log_prob, indexes = torch.topk(decoder_output, K - len(completed_hypotheses))

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((decoder_hidden[0][:, live_id, :].unsqueeze(1),
                                           decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                                          live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.itos[n.wordid.item()])
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    token = self.vocab.itos[n.wordid.item()]
                    if token != '<s>':
                        utterance.append(self.vocab.itos[n.wordid.item()])
                    else:
                        break

                utterance = utterance[::-1]

                utterances.append(utterance)

                # only save the top 1
                break

            decoded_batch.append(utterances[0])

        decoded_batch = [' '.join(sent) for sent in decoded_batch]

        return decoded_batch

    @torch.no_grad()
    def greedy_decode(self, z, max_length=30):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size, nz = z.size()
        decoded_batch = [[] for _ in range(batch_size)]

        # (1, batch_size, nz)
        c_init = self.linear_in(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size,
                                     dtype=torch.long, device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < max_length:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = word_embed.squeeze(2)
            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.linear_out(output)
            output_logits = decoder_output.squeeze(0)

            # (batch_size)
            max_index = torch.argmax(output_logits, dim=1)

            decoder_input = max_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.itos[max_index[i].item()])

            mask = (max_index != end_symbol) * mask

        decoded_batch = [' '.join(sent) for sent in decoded_batch]

        return decoded_batch

    @torch.no_grad()
    def sample_decode(self, z, max_length=30):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size, nz = z.size()
        decoded_batch = [[] for _ in range(batch_size)]

        # (1, batch_size, nz)
        c_init = self.linear_in(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < max_length:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = word_embed.squeeze(2)
            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.linear_out(output)
            output_logits = decoder_output.squeeze(0)

            # (batch_size)
            sample_prob = F.softmax(output_logits, dim=1)
            sample_index = torch.multinomial(sample_prob, num_samples=1).squeeze()

            decoder_input = sample_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.itos[sample_index[i].item()])

            mask = (sample_index != end_symbol) * mask

        decoded_batch = [' '.join(sent) for sent in decoded_batch]

        return decoded_batch

class lm_VAE(pl.LightningModule):

    def __init__(self, vocab, embedding_dims, hidden_dims, latent_dims, z_iters, aggressive, inner_iter, kl_weight_start, anneal_rate, decoding_strategy):

        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.z_iters = z_iters
        self.anneal_rate = anneal_rate
        self.kl_weight = kl_weight_start
        self.aggressive = aggressive
        self.inner_iter = inner_iter
        self.decoding_strategy = decoding_strategy

        self.encoder = LSTM_Encoder(vocab=self.vocab,
                                    embedding_dims=self.embedding_dims,
                                    hidden_dims=hidden_dims,
                                    latent_dims=self.latent_dims)

        self.decoder = LSTM_Decoder(vocab=self.vocab,
                                    embedding_dims=self.embedding_dims,
                                    hidden_dims=hidden_dims,
                                    latent_dims=self.latent_dims)

        vocab_mask = torch.ones(len(self.vocab))
        self.loss_fn = nn.CrossEntropyLoss(weight=vocab_mask, reduction='none')

    def forward(self, batch):

        text, _ = batch.text, batch.label

        mean, log_std = self.encoder(text)

        MI = self.encoder.MutualInformation(text,
                                            stats=(mean, log_std),
                                            z_iters=self.z_iters)

        L_reg = KLD(mean, log_std)

        z = sample_reparameterize(mean, log_std)

        target = text[1: , :]
        logits = self.decoder.forward(text[:-1, :], z)

        L_rec = self.loss_fn(logits.view(-1, logits.size(2)),
                             target.contiguous().view(-1))
        L_rec = torch.sum(L_rec.view(logits.size(1), -1), dim=-1)

        bpd = ELBO_to_BPD(L_rec+L_reg, text.size())

        return torch.mean(L_rec), torch.mean(L_reg), MI, torch.mean(bpd)

    def configure_optimizers(self):
        encoder_opt = optim.SGD(self.encoder.parameters(), lr=1.0, momentum=0)
        decoder_opt = optim.SGD(self.decoder.parameters(), lr=1.0, momentum=0)

        enc_sched = optim.lr_scheduler.ReduceLROnPlateau(encoder_opt, factor=0.5, patience=2, min_lr=1e-3,
                                                         verbose=True)
        enc_sched_dict = {
            'scheduler': enc_sched,
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'Valid BPD',
            'strict': True
            }

        dec_sched = optim.lr_scheduler.ReduceLROnPlateau(decoder_opt, factor=0.5, patience=2, min_lr=1e-3,
                                                         verbose=True)
        dec_sched_dict = {
            'scheduler': dec_sched,
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'Valid BPD',
            'strict': True
            }

        return [encoder_opt, decoder_opt], [enc_sched_dict, dec_sched_dict]

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch > 5:
            self.aggressive = False

        (encoder_opt, decoder_opt) = self.optimizers()

        if self.aggressive:

            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            for i in range(self.inner_iter):
                L_rec, L_reg, MI, bpd = self.forward(batch)

                loss = L_rec + self.kl_weight * L_reg

                self.log("Train - Inner Encoder MI", MI)
                self.log("Train - Inner ELBO", loss)
                self.log("Train - Inner BPD", bpd)

                burn_sents_len, burn_batch_size = batch.text.size()
                burn_num_words += (burn_sents_len - 1) * burn_batch_size
                burn_cur_loss += loss.sum().detach()

                self.manual_backward(loss, encoder_opt)
                encoder_opt.step()

                if i % 5 == 0:
                    print('Inner:', loss)
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        self.log("Train - Inner Steps", i)
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

        L_rec, L_reg, MI, bpd = self.forward(batch)

        loss = L_rec + self.kl_weight * L_reg
        print('Outer:', loss)

        self.log("Train - Outer L_rec", L_rec,
                 on_step=True, on_epoch=False)
        self.log("Train - Outer L_reg", L_reg,
                 on_step=True, on_epoch=False)
        self.log("Train - Outer Encoder MI", MI,
                 on_step=True, on_epoch=False)
        self.log("Train - Outer ELBO", loss,
                 on_step=True, on_epoch=False)
        self.log("Train - Outer BPD", bpd,
                 on_step=True, on_epoch=False, prog_bar=True)
        self.log("Train - Outer KLD Weight", self.kl_weight,
                 on_step=True, on_epoch=False)

        if not self.aggressive:
            self.manual_backward(loss, encoder_opt)
            encoder_opt.step()

        self.manual_backward(loss, decoder_opt)
        decoder_opt.step()

        self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)

    def validation_step(self, batch, batch_idx):

        L_rec, L_reg, MI, bpd = self.forward(batch)

        self.log("Valid Reconstruction Loss", L_rec,
                 on_epoch=False)
        self.log("Valid Regularization Loss", L_reg,
                 on_epoch=True)
        self.log("Valid Encoder MI", MI,
                 on_epoch=True)
        self.log("Valid BPD", bpd,
                 on_epoch=True)

        loss = L_rec + self.kl_weight * L_reg

        self.log("Valid ELBO", loss,
                 on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        L_rec, L_reg, MI, bpd = self.forward(batch)

        self.log("Test Reconstruction Loss", L_rec,
                 on_step=True, on_epoch=False)
        self.log("Test Regularization Loss", L_reg,
                 on_step=True, on_epoch=False)
        self.log("Test Encoder MI", MI,
                 on_step=True, on_epoch=False)
        self.log("Test BPD", bpd,
                 on_step=True, on_epoch=False)

        loss = L_rec + self.kl_weight * L_reg

        self.log("Test ELBO", loss,
                 on_step=True, on_epoch=False)

    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
    #                   on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #    # Update rules for the encoder
    #    if epoch > 5:
    #        self.aggressive = False
    #
    #    if self.aggressive:
    #        if optimizer_idx == 0 and batch_idx <= self.inner_iter:
    #            optimizer.step(closure=optimizer_closure)
    #
    #        elif optimizer_idx == 1 and batch_idx > self.inner_iter:
    #            optimizer.step(closure=optimizer_closure)
    #
    #    else:
    #        if optimizer_idx == 0:
    #           optimizer.step(closure=optimizer_closure)
    #        elif optimizer_idx == 1:
    #            optimizer.step(closure=optimizer_closure)

    @torch.no_grad()
    def decode(self, text, decoding_strategy=None, beam_length=5):

        mean, log_std = self.encoder(text)
        z = sample_reparameterize(mean, log_std)
        z = z.squeeze()

        if decoding_strategy == None:
            decoding_strategy = self.decoding_strategy

        if decoding_strategy == 'beam_search':
            text_sample = self.decoder.beam_search_decode(z, beam_length)
        elif decoding_strategy == 'greedy':
            text_sample = self.decoder.greedy_decode(z)
        elif decoding_strategy == 'sample':
            text_sample = self.decoder.sample_decode(z)

        return text_sample

    @torch.no_grad()
    def latent_sweep(self, text, zi, num=7, decoding_strategy=None, beam_length=5):

        if len(text.size()) >= 1:
            input = text[:, 0]
        else:
            input = text

        mean, log_std = self.encoder(input.unsqueeze(1))
        z_init = sample_reparameterize(mean, log_std).squeeze()

        vals = np.linspace(-3, 3, num=num)
        sweep = []
        for val in vals:
            z_val = z_init
            z_val[zi] += val

            sweep.append(z_val)

        z = torch.stack(sweep, dim=0)

        if decoding_strategy == None:
            decoding_strategy = self.decoding_strategy

        if decoding_strategy == 'beam_search':
            text_sample = self.decoder.beam_search_decode(z, beam_length)
        elif decoding_strategy == 'greedy':
            text_sample = self.decoder.greedy_decode(z)
        elif decoding_strategy == 'sample':
            text_sample = self.decoder.sample_decode(z)

        sweep_text = []
        for n, sent in enumerate(text_sample):
            sweep_text.append(['%+4.2f' % (vals[n]), sent])

        return sweep_text

import math
import time

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from utils.lagging_encoder import *
from utils.vae_loss import *

class LSTM_Encoder(nn.Module):

    def __init__(self, vocab, latent_dims, embedding_dims, n_layers, hidden_dims):
        super(LSTM_Encoder, self).__init__()

        self.vocab = vocab
        self.latent_dims = latent_dims
        self.embedding_dims = embedding_dims
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.layer_dim = n_layers * hidden_dims

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=self.embedding_dims,
                                      padding_idx=vocab['<pad>'])

        self.lstm = nn.LSTM(input_size=self.embedding_dims,
                            hidden_size=self.hidden_dims,
                            num_layers=self.n_layers,
                            batch_first=False)

        self.mean = nn.Linear(in_features=2*self.layer_dim, out_features=self.latent_dims)
        self.log_std = nn.Linear(in_features=2*self.layer_dim, out_features=self.latent_dims)

        self.reset_parameters()

    def reset_parameters(self, std=0.01):
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        nn.init.uniform_(self.embedding.weight, -std * 10, std * 10)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input):

        # (seq_len, batch_size, embedding_dim)
        embeddings = self.embedding(input)

        mean_, log_std_ = [], []
        for t, token in enumerate(embeddings):

            # (n_layers, batch_size, hidden_dim) x2
            _, (h_t, c_t) = self.lstm(token.unsqueeze(0), None if t == 0 else (h_t, c_t))

            # (batch_size, n_layers * hidden_dim * 2)
            hidden_out = torch.cat([h_t, c_t], dim=2).contiguous().view(token.size(0), -1)

            # (batch_size, n_layers * hidden_dim * 2) -> (batch_size, n_layers * hidden_dim * 2)
            mean_.append(self.mean(hidden_out))
            log_std_.append(self.log_std(hidden_out))

        mean_, log_std_ = torch.stack(mean_), torch.stack(log_std_)

        return mean_, log_std_

    @torch.no_grad()
    def mi_input_latent(self, x, stats=None, z_iters=1, debug=False):
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

        x_batch, z_dim = mean.size(0), mean.size(1)
        z_batch = x_batch * z_iters

        # E_{q(z|x)}log(q(z|x)) = -0.5*(K+L)*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * z_dim * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, z_dim]
        z_samples = []
        for i in range(z_iters):
            z_samples.append(sample_reparameterize(mean, logstd).unsqueeze(1))
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

    def __init__(self, vocab, latent_dims, embedding_dims, n_layers, hidden_dims, dropout, teacher_force_p):
        super(LSTM_Decoder, self).__init__()

        self.vocab = vocab
        self.latent_dims = latent_dims
        self.embedding_dims = embedding_dims
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.layer_dim = n_layers * hidden_dims
        self.teacher_force_p = teacher_force_p

        self.linear_in = nn.Sequential(nn.Linear(in_features=latent_dims,
                                                 out_features=self.layer_dim),
                                       nn.Dropout(p=dropout)
                                       )

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=embedding_dims,
                         padding_idx=vocab['<pad>'])

        self.lstm = nn.LSTM(input_size=embedding_dims + self.hidden_dims,
                       hidden_size=hidden_dims,
                       num_layers=n_layers,
                       batch_first=False)

        self.linear_out = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(in_features=hidden_dims, out_features=len(vocab))
                                        )

        self.reset_parameters()

    def reset_parameters(self, std=0.01):
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        nn.init.uniform_(self.embedding.weight, -std * 10, std * 10)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z, text=None, max_length=82):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        z_ = self.linear_in(z)

        # (n_layers, batch_size, hidden_dim)
        z_ = z_.view(self.n_layers, z.size(0), self.hidden_dims)

        # (n_layers, batch_size, hidden_dim)
        c = z_
        h = torch.tanh(c)

        input = torch.full(size=(1, z.size(0)), fill_value=self.vocab['<s>'],
                           dtype=torch.long, device=self.device)
        logits = []
        for t in range(max_length - 1):
            # (1, batch_size, embedding_dim)
            embedded_dec = self.embedding(input)

            # (1, batch_size, embedding_dim + hidden_dim)
            embedded_dec_a = torch.cat([embedded_dec, z_[-1, :, :].unsqueeze(0)], dim=-1)

            # (1, batch_size, hidden_dim)
            output, (h, c) = self.lstm(embedded_dec_a, (h, c))

            # (1, batch_size, vocab_size)
            logits_t = self.linear_out(output)

            teacher_force = torch.rand((1)) < self.teacher_force_p
            if teacher_force and text != None:
                input = text[t + 1].unsqueeze(0)
            else:
                input = torch.argmax(logits_t, dim=-1)

            logits.append(logits_t)

        # (seq_len, batch_size, vocab_size)
        logits = torch.cat(logits, dim=0)

        return logits

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
        """

        z_ = self.linear_in(z)

        # (n_layers, batch_size, hidden_dim)
        z_ = z_.view(self.n_layers, z.size(0), self.hidden_dims)

        # (n_layers, batch_size, hidden_dim)
        c = z_
        h = torch.tanh(c)

        input = torch.full(size=(1, z.size(0)), fill_value=self.vocab['<s>'],
                           dtype=torch.long, device=self.device)
        tokens = []
        for t in range(max_length - 1):
            # (1, batch_size, embedding_dim)
            embedded_dec = self.embedding(input)

            # (1, batch_size, embedding_dim + hidden_dim)
            embedded_dec_a = torch.cat([embedded_dec, z_[-1, :, :].unsqueeze(0)], dim=-1)

            # (1, batch_size, hidden_dim)
            output, (h, c) = self.lstm(embedded_dec_a, (h, c))

            # (1, batch_size, vocab_size)
            logits_t = self.linear_out(output)

            input = torch.argmax(logits_t, dim=-1)
            tokens.append(input)

        # (seq_len, batch_size)
        tokens = torch.cat(tokens)

        decoded_batch = []
        for i in range(tokens.size(1)):
            sent = tokens[:,i]
            decoded_sent = [self.vocab.itos[sent[i].item()] for i in range(tokens.size(0))]
            decoded_batch.append(' '.join(decoded_sent))

        return decoded_batch

    @torch.no_grad()
    def sample_decode(self, z, tau=1, max_length=30):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        """greedy decoding from z
        Args:
            z: (batch_size, nz)
        """

        z_ = self.linear_in(z)

        # (n_layers, batch_size, hidden_dim)
        z_ = z_.view(self.n_layers, z.size(0), self.hidden_dims)

        # (n_layers, batch_size, hidden_dim)
        c = z_
        h = torch.tanh(c)

        input = torch.full(size=(1, z.size(0)), fill_value=self.vocab['<s>'],
                           dtype=torch.long, device=self.device)
        tokens = []
        for t in range(max_length - 1):
            # (1, batch_size, embedding_dim)
            embedded_dec = self.embedding(input)

            # (1, batch_size, embedding_dim + hidden_dim)
            embedded_dec_a = torch.cat([embedded_dec, z_[-1, :, :].unsqueeze(0)], dim=-1)

            # (1, batch_size, hidden_dim)
            output, (h, c) = self.lstm(embedded_dec_a, (h, c))

            # (1, batch_size, vocab_size)
            logits_t = self.linear_out(output)

            sample_prob = F.softmax(tau * logits_t, dim=-1)
            input = torch.multinomial(sample_prob.squeeze(), num_samples=1).permute(1, 0)

            tokens.append(input)

        # (seq_len, batch_size)
        tokens = torch.cat(tokens)

        decoded_batch = []
        for i in range(tokens.size(1)):
            sent = tokens[:, i]
            decoded_sent = [self.vocab.itos[sent[i].item()] for i in range(tokens.size(0))]
            decoded_batch.append(' '.join(decoded_sent))

        return decoded_batch

class text_VAE(pl.LightningModule):

    def __init__(self, vocab, latent_dims, embedding_dims, n_layers, hidden_dims, dropout, teacher_force_p, lr, decoding_strategy, aggressive, inner_iter, kl_weight_start, anneal_rate):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.latent_dims = latent_dims
        self.embedding_dims = embedding_dims
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.teacher_force_p = teacher_force_p
        self.lr = lr
        self.decoding_strategy = decoding_strategy

        # Aggressive training
        self.aggressive = aggressive
        self.inner_iter = inner_iter
        self.kl_weight = kl_weight_start
        self.anneal_rate = anneal_rate

        self.encoder = LSTM_Encoder(vocab, latent_dims, embedding_dims, n_layers, hidden_dims)
        self.decoder = LSTM_Decoder(vocab, latent_dims, embedding_dims, n_layers, hidden_dims, dropout, teacher_force_p)

        self.t0 = time.time()

    def forward(self, batch):

        text, _ = batch.text, batch.label

        mean, log_std = self.encoder(text)

        z = sample_reparameterize(mean, log_std)

        logits = self.decoder(z[-1], text, max_length=z.size(0))

        return logits, mean, log_std

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch > 4:
            self.aggressive = False

        if self.global_step == 0:
            for schedule_dict in self.trainer.lr_schedulers:
                schedule_dict['scheduler'].cooldown_counter = 14

        (encoder_opt, decoder_opt) = self.optimizers()

        i=0
        if self.aggressive:

            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            for i in range(self.inner_iter):
                logits, mean, log_std = self.forward(batch)

                predict = logits.view(-1, logits.size(-1))
                targets = batch.text[1:].view(-1)

                L_rec = F.cross_entropy(predict, targets, reduction='mean',
                                        ignore_index=self.vocab['<pad>'])
                L_reg = torch.mean(KLD(mean, log_std))

                loss = L_rec + self.kl_weight * L_reg

                burn_sents_len, burn_batch_size = batch.text.size()
                burn_num_words += (burn_sents_len - 1) * burn_batch_size
                burn_cur_loss += loss.sum().detach()

                self.manual_backward(loss, encoder_opt)
                encoder_opt.step()

                if (i+1) % (self.inner_iter//5) == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        self.log("Train Inner Steps", i)
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

        logits, mean, log_std = self.forward(batch)

        predict = logits.view(-1, logits.size(-1))
        targets = batch.text[1:].view(-1)

        L_rec = F.cross_entropy(predict, targets, reduction='mean',
                                ignore_index=self.vocab['<pad>'])
        L_reg = torch.mean(KLD(mean, log_std))

        loss = L_rec + self.kl_weight * L_reg

        if not self.aggressive:
            self.manual_backward(loss, encoder_opt, retain_graph=True)
        self.manual_backward(loss, decoder_opt)

        if not self.aggressive:
            encoder_opt.step()
        decoder_opt.step()


        self.log("Train L_rec", L_rec)
        self.log("Train L_reg", L_reg)
        self.log("Train ELBO", loss, prog_bar=True)
        self.log("Train KLD Weight", self.kl_weight)

        acc = (torch.argmax(predict, dim=-1).detach() == targets).float().mean()
        self.log('Train acc', acc)

        self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)

        self.trainer.train_loop.running_loss.append(loss)

        if batch_idx==0 or batch_idx % 5 == 0:
            t1 = time.time()
            dt = (t1 - self.t0) / 60
            mins, secs = int(dt), int((dt - int(dt)) * 60)

            print(f"Time: {mins:4d}m {secs:2d}s| Train {int(self.current_epoch):03d}.{int(batch_idx):03d}: L_rec={L_rec:6.2f}, L_reg={L_reg:6.2f}, Inner iters={int(i):02d}, KL weight={self.kl_weight:4.2f}, Acc={acc:.2f}")

    def validation_step(self, batch, batch_idx):

        text = batch.text

        logits, mean, log_std = self.forward(batch)

        predict = logits.view(-1, logits.size(2))
        targets = text[1:].view(-1)

        L_rec = F.cross_entropy(predict, targets, reduction='mean', ignore_index=self.vocab['<pad>'])
        L_reg = torch.mean(KLD(mean, log_std))

        elbo = L_rec + L_reg

        self.log('Valid ELBO', elbo)
        self.log('Valid L_rec', L_rec)
        self.log('Valid L_reg', L_reg)

        acc = (torch.argmax(predict, dim=-1).detach() == targets).float().mean()
        self.log('Valid acc', acc)

        mi = self.encoder.mi_input_latent(text, stats=(mean[-1], log_std[-1]), z_iters=10)
        self.log('Encoder MI', mi)

        t1 = time.time()
        dt = (t1 - self.t0) / 60
        mins, secs = int(dt), int((dt - int(dt)) * 60)

        print(f"Time: {mins:4d}m {secs:2d}s| Valid {int(self.current_epoch):03d}.{int(batch_idx):03d}: L_rec={L_rec:6.2f}, L_reg={L_reg:6.2f}, Acc={acc:.2f}, MI={acc:5.2f}")

    def test_step(self, batch, batch_idx):

        text = batch.text

        logits, mean, log_std = self.forward(batch)

        predict = logits.view(-1, logits.size(2))
        targets = text[1:].view(-1)

        L_rec = F.cross_entropy(predict, targets, reduction='mean', ignore_index=self.vocab['<pad>'])
        L_reg = torch.mean(KLD(mean, log_std))

        elbo = L_rec + L_reg

        self.log('Test ELBO', elbo)
        self.log('Test L_rec', L_rec)
        self.log('Test L_reg', L_reg)

        acc = (torch.argmax(predict, dim=-1).detach() == targets).float().mean()
        self.log('Test acc', acc)


    #def configure_optimizers(self):
    #    optimizer = optim.AdamW(self.parameters(), lr=self.lr, eps=1e-6, weight_decay=1e-4)
    #
    #    return [optimizer]

    def configure_optimizers(self):
        encoder_opt = optim.SGD(self.encoder.parameters(), lr=1.0, momentum=0)
        decoder_opt = optim.SGD(self.decoder.parameters(), lr=1.0, momentum=0)

        enc_sched = optim.lr_scheduler.ReduceLROnPlateau(encoder_opt, factor=0.5, patience=1, min_lr=1e-3,
                                                        verbose=True)
        enc_sched_dict = {
            'scheduler': enc_sched,
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'Valid ELBO',
            'strict': True
        }

        dec_sched = optim.lr_scheduler.ReduceLROnPlateau(decoder_opt, factor=0.5, patience=1, min_lr=1e-3,
                                                        verbose=True)
        dec_sched_dict = {
            'scheduler': dec_sched,
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'Valid ELBO',
            'strict': True
        }

        return [encoder_opt, decoder_opt], [enc_sched_dict, dec_sched_dict]

    @torch.no_grad()
    def reconstruct(self, text, decoding_strategy=None, beam_length=5):

        mean, _ = self.encoder(text)
        z = mean

        if decoding_strategy == None:
            decoding_strategy = self.decoding_strategy

        if decoding_strategy == 'beam_search':
            raise NotImplementedError()
            #text_sample = self.decoder.beam_search_decode(z, beam_length)
        elif decoding_strategy == 'greedy':
            text_sample = self.decoder.greedy_decode(z[-1])
        elif decoding_strategy == 'sample':
            text_sample = self.decoder.sample_decode(z[-1])

        return text_sample

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

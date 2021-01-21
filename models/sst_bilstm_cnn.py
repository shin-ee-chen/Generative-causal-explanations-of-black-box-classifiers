import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class sst_bilstm_cnn(pl.LightningModule):

    def __init__(self, vocab, dropout, lstm_hidden, filters, cnn_ksize, max_ksize, padded_length, M, lr, lr_decay):
        super().__init__()
        self.save_hyperparameters()

        self.dropout = dropout
        self.lstm_hidden = lstm_hidden
        self.filters = filters
        self.cnn_ksize = cnn_ksize
        self.max_ksize = max_ksize
        self.M = M
        self.lr = lr
        self.lr_decay = lr_decay

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=vocab.vectors.size(0), embedding_dim=vocab.vectors.size(1), padding_idx=vocab['<pad>']),
            nn.Dropout(p=self.dropout[0])
            )
        self.embedding[0].weight.data.copy_(vocab.vectors)
        self.embedding[0].weight.requires_grad = False

        self.lstm = nn.Sequential(
            nn.LSTM(input_size=vocab.vectors.size(1), hidden_size=self.lstm_hidden, bidirectional=True),
            nn.Dropout(p=self.dropout[1])
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=self.cnn_ksize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_ksize),
            nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.cnn_ksize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_ksize),
            nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.cnn_ksize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_ksize),
        )

        height, width = padded_length, self.lstm_hidden
        for i in range(3):
            height = int(((height - self.cnn_ksize + 1) - self.max_ksize)/ self.max_ksize + 1)
            width = int(((width - self.cnn_ksize + 1) - self.max_ksize) / self.max_ksize + 1)
        print('Output shape of convolutions:', height, width)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout[2]),
            nn.Linear(in_features=self.filters * height * width, out_features=self.M)
        )

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, text):

        embedding = self.embedding(text)

        (h, c) = self.lstm[0](embedding)
        h = h.view(text.size(0), text.size(1), 2, self.lstm_hidden)
        rnn_output = self.lstm[1]((h[:, :, 0, :] + h[:, :, 1, :]).permute(1, 0, 2).unsqueeze(1))

        feature_maps = self.conv(rnn_output)

        logits = self.linear(feature_maps)

        return logits

    def configure_optimizers(self):

        optimizer = optim.AdamW([
            {'params': self.embedding.parameters()},
            {'params': self.lstm.parameters()},
            {'params': self.conv.parameters()},
            {'params': self.linear.parameters()}
            ], self.lr, weight_decay=1e-2)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        loss = self.loss_module(logits, labels)

        preds = logits.argmax(dim=-1).detach()
        acc = (preds == labels).float().mean()

        self.log('Train acc', acc, on_step=True)
        self.log('Train loss', loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        loss = self.loss_module(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('Valid loss', loss, on_epoch=True)
        self.log('Valid acc', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        preds = logits.argmax(dim=-1).detach()
        acc = (labels == preds).float().mean()

        self.log('Test acc', acc, on_epoch=True)

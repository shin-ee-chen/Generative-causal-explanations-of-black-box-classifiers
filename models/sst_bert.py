import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
# from transformers import BertConfig, BertForSequenceClassification
from transformers import SqueezeBertConfig, SqueezeBertForSequenceClassification

class SST_Bert_clf(pl.LightningModule):

    def __init__(self, M, dropout, lr, layer_decay, warm_up, max_lr_rate, steps_per_cycle):
        super().__init__()
        self.save_hyperparameters()

        self.M = M
        self.dropout = dropout
        self.lr = lr
        self.layer_decay = layer_decay
        self.warm_up = warm_up
        self.max_lr_rate = max_lr_rate
        self.steps_per_cycle = steps_per_cycle

        config = SqueezeBertConfig.from_pretrained('squeezebert/squeezebert-mnli-headless', num_labels=self.M,
                                            return_dict=True, gradient_checkpointing=True)

        self.model = SqueezeBertForSequenceClassification.from_pretrained('squeezebert/squeezebert-mnli-headless')
        self.model.dropout = nn.Dropout(p=self.dropout)
        self.model.classifier = nn.Linear(in_features=self.model.config.hidden_size,
                                          out_features=self.M)

        for layer in list(self.model.transformer.encoder.layers.children())[0:10]:
            layer.requires_grad = False

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, text):
        logits = self.model(text)['logits']

        return logits

    def configure_optimizers(self):

        per_layer = [{'params': layer.parameters(), 'lr': self.layer_decay**(12 - n) * self.lr} for
                     n, layer in enumerate(list(self.model.transformer.encoder.layers.children())[10:])]
        per_layer = [{'params': self.model.transformer.pooler.parameters()}]
        per_layer += [{'params': self.model.classifier.parameters()}]

        optimizer = optim.Adam(per_layer, lr=self.lr)

        step_size_up = int((self.warm_up * self.steps_per_cycle))
        step_size_down = self.steps_per_cycle - step_size_up

        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                base_lr = self.lr,
                                                step_size_up=step_size_up,
                                                step_size_down=step_size_down,
                                                mode='triangular',
                                                max_lr=32 * self.lr,
                                                cycle_momentum=False)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        self.model.train()

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        loss = self.loss_module(logits, labels)

        preds = logits.argmax(dim=-1).detach()
        acc = (preds == labels).float().mean()

        self.log('Train acc', acc)
        self.log('Train loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        self.model.eval()

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        preds = logits.argmax(dim=-1).detach()
        acc = (labels == preds).float().mean()

        self.log('Valid acc', acc)
        print('Valid acc:', acc)

    def test_step(self, batch, batch_idx):

        self.model.eval()

        text, labels = batch.text, batch.label

        logits = self.forward(text)
        preds = logits.argmax(dim=-1).detach()
        acc = (labels == preds).float().mean()

        self.log('Test acc', acc)

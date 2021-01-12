import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

class CNN_OShaugnessy(nn.Module):

    def __init__(self, M=2):
        """ The MNIST CNN classifier introduced by OShaugnessy et al. (see Table 1, Appendix D).

        Args:
        - img_channels: int, the number of channels the MNIST images have
        - M: int, the number of output nodes
            
        """
        super().__init__()

        self.img_channels = 1
        self.M = M

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=32,
                      kernel_size=3, stride=1, padding=0),          # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=0),          # 26x26 -> 24x24
            nn.ReLU(),
            # 24x24 -> 23x23
            nn.MaxPool2d(kernel_size=2),

            # Linear Layer
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=self.M)
        )

    def forward(self, X):
        Y = self.net(X)

        return Y

class MNIST_CNN(pl.LightningModule):

    def __init__(self, model_param_set, M, lr, momentum):
        super().__init__()
        self.save_hyperparameters()

        self.M = M
        self.lr = lr
        self.momentum = momentum

        if model_param_set == 'OShaugnessy':
            self.model = CNN_OShaugnessy(self.M)
        else:
            print('Unknown classifier parameter set.')
            return None

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=self.momentum)

        # TODO Find used implementation of learning rate decay
        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[100, 150], gamma=0.1)

        return [optimizer]  # , [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('Train_acc', acc, on_step=False, on_epoch=True)
        self.log('Train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('Valid_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('Test_acc', acc)

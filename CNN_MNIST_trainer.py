import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
# TODO uncomment LearningRateMonitor once properly implemented
from pytorch_lightning.callbacks import ModelCheckpoint#, LearningRateMonitor

import tqdm

from MNIST import MNIST_limited
from CNN import CNN_OShaugnessy

# ! This should be handled using arg-parser
CHECKPOINT_PATH = './Models/MNIST_CNN'

# ! This should be handled using arg-parser
device = 'cuda:0'

class MNIST_CNN_OShaugnessy(pl.LightningModule):

    def __init__(self):
        """
        
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters()

        self.model = CNN_OShaugnessy()
                
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros(
            (1, 1, 28, 28), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):     
        # ! This should be handled using arg-parser
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.5)
        
        # TODO Find used implementation of learning rate decay 
        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[100, 150], gamma=0.1)
        
        return [optimizer]#, [scheduler]

    def training_step(self, batch, batch_idx):

        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('Train_acc', acc, on_step=False, on_epoch=True)
        self.log('Train_loss', loss)
        return loss  # Return tensor to call ".backward" on

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

def train_model(save_name=None, **kwargs):
    """
    Inputs:
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    
    train_set, valid_set = MNIST_limited(train=True)
    test_set = MNIST_limited(train=False)

    # ! The batchsize should be handled using arg-parser
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True,
                                drop_last=True, pin_memory=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=64, shuffle=False,
                                drop_last=True, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False,
                                drop_last=True, pin_memory=True, num_workers=0)
    
    # ! Max epochs needs to be handled by arg parser.
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name), # Where to save models
                         # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="max", monitor="Valid_acc"),
                         # We run on a single GPU (if possible)
                         gpus=1 if str(device) == "cuda:0" else 0,
                         # How many epochs to train for if no patience is set
                         max_epochs=30,
                         # Log learning rate every epoch
                         #callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1)                                                                
    
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = MNIST_CNN_OShaugnessy.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = MNIST_CNN_OShaugnessy()
        trainer.fit(model, train_loader, valid_loader)
        model = MNIST_CNN_OShaugnessy.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(
        model, test_dataloaders=valid_loader, verbose=False)
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=False)
    result = {"Test": test_result[0]["Test_acc"],
              "Valid": val_result[0]["Valid_acc"]}

    return model, result

if __name__ == 'main':
    CNN_MNIST_model, CNN_MNIST_results = train_model(save_name='MNIST_CNN')

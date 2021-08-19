import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


class CustomDataset(Dataset):
    def __init__(self, dir_x, dir_y):
        self.x = torch.from_numpy(idx2numpy.convert_from_file(dir_x).astype(np.float32))
        self.y = torch.from_numpy(idx2numpy.convert_from_file(dir_y).astype(np.int64))

        # Normalize x
        self.x = self.x / 255

        # Reshape x
        self.x = self.x.view(-1, 1, 28, 28)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        # MNIST images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer_3 = nn.Linear(32 * 26 * 26, 64)
        self.layer_4 = nn.Linear(64, 64)
        self.layer_5 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # input: (b, 1, 28, 28)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer_2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.layer_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        y = F.log_softmax(x, dim=1)

        # output (b, 10)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = pl.metrics.Accuracy()(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_acc', accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = pl.metrics.Accuracy()(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def test_on_dataset(dataset, model, index):
    # Test a model on an image of a dataset

    x = dataset[index][0]
    logits = model(x.view(1, 1, 28, 28))
    y_hat = torch.argmax(logits, dim=1)
    y_true = dataset[index][1]

    plt.imshow(x.view(28, 28))
    print('Prediction:', y_hat)
    print('Truth:', y_true)


def get_accuracy(dataset, model):
    # Test a model on a dataset and return the accuracy
    if isinstance(dataset, torch.utils.data.dataset.Subset):
        x = dataset.dataset.x[dataset.indices]
        y = dataset.dataset.y[dataset.indices]
    else:
        x = dataset.x
        y = dataset.y

    model.eval()
    logits = model(x.view(-1, 1, 28, 28))
    y_hat = torch.argmax(logits, dim=1)
    accuracy = pl.metrics.Accuracy()(y_hat, y)

    print('Accuracy:', accuracy)

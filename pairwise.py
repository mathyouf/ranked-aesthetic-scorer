import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop
import torch.nn.functional as F
import torch
from transformers import AutoProcessor, AutoModel

class AestheticScoreMLP1(pl.LightningModule):
    def __init__(self, input_size, x1col='emb1', x2col='emb2', ycol='label'):
        super().__init__()
        self.input_size = input_size
        self.x1col = x1col
        self.x2col = x2col
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # Add ReLU?
            #nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, emb1, emb2):
        x1 = self.layers(emb1)
        x2 = self.layers(emb2)
        x = torch.cat((x1, x2), dim=1)
        x = x / 10
        # clamp to 0-1
        x = torch.clamp(x, 0, 1)
        # Subtract x[:, 0] by x[:, 1]
        x = x[:, 0] - x[:, 1]
        x = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x))
        return x

    def training_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.forward(x1, x2)
        loss = F.binary_cross_entropy(x_hat, y)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1).flatten()
        x_hat = self.forward(x1, x2)
        loss = F.binary_cross_entropy(x_hat, y)
        self.log("valid/loss_epoch", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

class AestheticScoreMLP2(pl.LightningModule):
    def __init__(self, input_size, x1col='emb1', x2col='emb2', ycol='label'):
        super().__init__()
        self.input_size = input_size
        self.x1col = x1col
        self.x2col = x2col
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # Add ReLU?
            #nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, emb1, emb2):
        x1 = self.layers(emb1)
        x2 = self.layers(emb2)
        x = torch.cat((x1, x2), dim=1)
        x = x / 10
        # clamp to 0-1
        x = torch.clamp(x, 0, 1)
        # Convert distribution to 0-inf using -log(1-x)
        x = -torch.log(1 - x)
        # Apply softmax 
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1)
        # Convert y (8, 1) to y_vec (8, 2) where y_vec[:, 0] = y and y_vec[:, 1] = 1 - y
        y_vec = torch.cat((y, 1 - y), dim=1)
        x_hat = self.forward(x1, x2)
        loss = F.binary_cross_entropy(x_hat, y_vec)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1)
        # Convert y (8, 1) to y_vec (8, 2) where y_vec[:, 0] = y and y_vec[:, 1] = 1 - y
        y_vec = torch.cat((y, 1 - y), dim=1)
        x_hat = self.forward(x1, x2)
        loss = F.binary_cross_entropy(x_hat, y_vec)
        self.log("valid/loss_epoch", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
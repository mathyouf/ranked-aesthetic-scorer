import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop
import torch.nn.functional as F
import torch
from transformers import AutoProcessor, AutoModel

# Could we instead predict the score and stddev using this method: https://github.com/yunxiaoshi/Neural-IMage-Assessment?
# instead of classifying images to low/high score
# or regressing to the mean score, the distribution of ratings are
# predicted as a histogram. To this end, we use the squared EMD
# (earth mover’s distance) loss proposed in [21], which shows
# a performance boost in classification with ordered classes.
# Our experiments show that this approach also leads to more
# accurate prediction of the mean score. Also, as shown in
# aesthetic assessment case [1], non-conventionality of images
# is directly related to score standard deviations. Our proposed
# paradigm allows for predicting this metric as well.

class MLPOriginal(pl.LightningModule):
    # From: https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/train_predictor.py
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class AestheticScoreMLP0(pl.LightningModule):
    def __init__(self, input_size, x1col='emb1', x2col='emb2', ycol='label'):
        super().__init__()
        self.input_size = input_size
        self.x1col = x1col
        self.x2col = x2col
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2), 
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, emb):
        x = self.layers(emb)
        return x

    def training_step(self, batch, batch_idx):
        # Get x values
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]

        # Get unnormalized aesthetic scores
        s1 = self(x1)
        s2 = self(x2)
        delta_s = s1 - s2

        # Get y values
        y_hat = torch.sigmoid(delta_s) # probability of x1 preferred x2  0.0-1.0
        y = batch[self.ycol].reshape(-1, 1) # observed probability

        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    #TODO
    def predict(self, x):
        unnormalized_score = self(x)
        # apply CDF normalization (sklearn empirical cumulative distribution function - could do in numpy) to get values 0-1
        # maybe use numpy - normalized histogra (np.historgram)
        score = False
        assert score
        return score

    def validation_step(self, batch, batch_idx):
        # Get x values
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]

        # Get unnormalized aesthetic scores
        s1 = self(x1)
        s2 = self(x2)
        delta_s = s1 - s2

        # Get y values
        y_hat = torch.sigmoid(delta_s) # probability of x1 preferred x2  0.0-1.0
        y = batch[self.ycol].reshape(-1, 1) # observed probability

        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train/loss', loss, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

#TODO: test this yo! thx charles
import numpy as np
def make_cdf(model, embs):    
    unnormalized_scores = model(embs)
    counts, edges = np.histogram(unnormalized_scores)  # check this is normalized to sum to 1
    probs = counts / np.sum(counts)
    ecdf = np.cumsum(probs)

    def run_cdf(unnormalized_score):
        change_point =  np.diff(np.int(unnormalized_score < edges))
        return np.dot(ecdf, change_point)

    return run_cdf

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
        # Requires grad
        x1 = self.layers(emb1)
        x2 = self.layers(emb2)
        # fix for: element 0 of tensors does not require grad and does not have a grad_fn
        x1 = x1.detach()
        x2 = x2.detach()
        x = torch.cat((x1, x2), dim=1)
        x = x / 10
        # clamp to 0-1
        x = torch.clamp(x, 0, 1)
        # Subtract x[:, 0] by x[:, 1]
        x = x[:, 0] - x[:, 1]
        return x

    def training_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1).flatten()
        # Convert 0 to -1
        y = 2 * y - 1
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
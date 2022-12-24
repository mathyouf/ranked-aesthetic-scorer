# This Model retrains the aesthetic model on a head which determines the higher ranked of two inputs #

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop
import torch.nn.functional as F
import torch
from transformers import AutoProcessor, AutoModel
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image
from dataset import RankDataModule
import requests

class AestheticScoreMLP(pl.LightningModule):
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
        # Create final layer from 0-1
        self.final_layer = nn.Sequential(
            # Sigmoid shifted to pivot at the value 5
            nn.Linear(2, 1),
        )

    def forward(self, emb1, emb2):
        x1 = self.layers(emb1)
        x2 = self.layers(emb2)
        x = torch.cat((x1, x2), dim=1)
        x = self.final_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.forward(x1, x2)
        loss = F.mse_loss(x_hat, y)
        self.log('train/loss', loss, on_epoch=True)
        # Try BCELoss
        # loss = F.binary_cross_entropy(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1 = batch[self.x1col]
        x2 = batch[self.x2col]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.forward(x1, x2)
        loss = F.mse_loss(x_hat, y)
        self.log("valid/loss_epoch", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.val_samples = val_samples
        print(self.val_samples)

    def on_validation_epoch_end(self, trainer, pl_module):
        emb1, emb2, label = self.val_samples['emb1'], self.val_samples['emb2'], self.val_samples['label']
        url1, url2 = self.val_samples['url1'], self.val_samples['url2']
        # Download images from urls reisze to 128x128
        imgs1 = [Image.open(requests.get(url, stream=True).raw).resize(size=(128, 128)) for url in url1]
        imgs2 = [Image.open(requests.get(url, stream=True).raw).resize(size=(128, 128)) for url in url2]
        caption1, caption2 = self.val_samples['caption1'], self.val_samples['caption2']
        # To device
        emb1 = emb1.to(pl_module.device)
        emb2 = emb2.to(pl_module.device)
        rank = pl_module(emb1, emb2)
        # Log the images - download url
         # Get image pairs and ranks
        table = wandb.Table(columns=['img1', 'img2', 'rank', 'label'])
        for i in range(len(imgs1)):
            img1 = wandb.Image(imgs1[i], caption=caption1[i])
            img2 = wandb.Image(imgs2[i], caption=caption2[i])
            # Round to 2 decimal places
            single_rank = round(rank[i].detach().cpu().item(), 2)
            # Convert to int
            single_label = int(label[i].detach().cpu().item())
            table.add_data(img1, img2, single_rank, single_label)
        # Log the table
        trainer.logger.experiment.log({
            "predictions": table
        })

# Folder which aggregates the results from running clip-retrieval on 1 or more datasets
tsv_embed_folders = ["./data/outputs/toloka/"]
roots = {"reddit": None, "tsv": tsv_embed_folders}
# Rank Dataset
dataset = RankDataModule(roots=roots, batch_size=8)
dataset.prepare_data()
dataset.setup()
samples = next(iter(dataset.val_dataloader()))
embedding_size = samples['emb1'].shape[-1]
# Aesthetic Model
model = AestheticScoreMLP(embedding_size)
# Weights from improved-aesthetic-scorer
s = torch.load("./model_weights/sac+logos+ava1-l14-linearMSE.pth")
model.load_state_dict(s, strict=False)
# Trainer for model + dataset
trainer = pl.Trainer(
    logger=WandbLogger(project="aesthetic-rankings"),    # W&B integration
    log_every_n_steps=50,   # set the logging frequency
    gpus=[3],               # Select GPUs
    max_epochs=10000,      # number of epochs
    deterministic=True,     # keep it deterministic
    callbacks=[ImagePredictionLogger(samples)]
)
trainer.fit(model, dataset)

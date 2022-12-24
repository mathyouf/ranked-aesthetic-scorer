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
            # Add ReLU?Oski x Dunk High SB 'Great White Shark'

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
embed_folder = "/home/matt/Desktop/Dataset-Creation-Curration/aesthetics/data/outputs/toloka/embeddings"
# Rank Dataset
dataset = RankDataModule(root=embed_folder, batch_size=8)
dataset.prepare_data()
dataset.setup()
samples = next(iter(dataset.val_dataloader()))
embedding_size = samples['emb1'].shape[-1]
# Aesthetic Model
model = AestheticScoreMLP(embedding_size)
# Weights from improved-aesthetic-scorer
s = torch.load("/home/matt/Desktop/Dataset-Creation-Curration/aesthetics/data/model_weights/sac+logos+ava1-l14-linearMSE.pth")
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

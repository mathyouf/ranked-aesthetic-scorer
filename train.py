# This Model retrains the aesthetic model on a head which determines the higher ranked of two inputs #

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
import requests
from datetime import datetime
from dataset import RankDataModule
from pairwise import AestheticScoreMLP0

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.val_samples = val_samples
        print(self.val_samples)

    def on_validation_epoch_end(self, trainer, pl_module):
        emb1, emb2, label = self.val_samples['emb1'], self.val_samples['emb2'], self.val_samples['label']
        url1, url2 = self.val_samples['url1'], self.val_samples['url2']
        # Download images from urls reisze to 128x128
        # Add try catch
        imgs1 = []
        imgs2 = []
        for url in url1:
            try:
                imgs1.append(Image.open(requests.get(url, stream=True).raw).resize((256, 256)))
            except:
                print("Error downloading image")
                imgs1.append(Image.new('RGB', (256, 256)))
        for url in url2:
            try:
                imgs2.append(Image.open(requests.get(url, stream=True).raw).resize((256, 256)))
            except:
                print("Error downloading image")
                imgs2.append(Image.new('RGB', (256, 256)))
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
            single_rank = int(rank[i].detach().cpu().item())
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
model = AestheticScoreMLP0(embedding_size)
# Weights from improved-aesthetic-scorer
s = torch.load("./model_weights/sac+logos+ava1-l14-linearMSE.pth")
model.load_state_dict(s, strict=False)
# Trainer for model + dataset
trainer = pl.Trainer(
    logger=WandbLogger(project="aesthetic-rankings"),    # W&B integration
    log_every_n_steps=50,   # set the logging frequency
    gpus=[3],               # Select GPUs
    max_epochs=100,      # number of epochs
    deterministic=True,     # keep it deterministic
    callbacks=[ImagePredictionLogger(samples)]
)
trainer.fit(model, dataset)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"./model_weights/{timestamp}.pth"
torch.save(model.state_dict(), save_path)
# This Model retrains the aesthetic model on a head which determines the higher ranked of two inputs #

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from PIL import Image
import urllib.request
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
        a_aes_def = self.val_samples['image_a_aes_default']
        b_aes_def = self.val_samples['image_b_aes_default']
        # Download images from urls reisze to 128x128
        # Add try catch
        imgs1 = []
        imgs2 = []
        img_size = 256
        for url in url1:
            try:
                img = urllib.request.urlretrieve(url)[0]
            except:
                print("Error downloading image")
            try:
                img = Image.open(img, mode='r')
            except:
                print("Error creating image")
            try:
                imgs1.append(img.resize((img_size, img_size)))
            except:
                print("Error resizing image")
                imgs1.append(Image.new('RGB', (img_size, img_size)))
        for url in url2:
            try:
                img = urllib.request.urlretrieve(url)[0]
            except:
                print("Error downloading image")
            try:
                img = Image.open(img, mode='r')
            except:
                print("Error creating image")
            try:
                imgs2.append(img.resize((img_size, img_size)))
            except:
                print("Error resizing image")
                imgs2.append(Image.new('RGB', (img_size, img_size)))

        caption1, caption2 = self.val_samples['caption1'], self.val_samples['caption2']

        # To device
        x1 = emb1.to(pl_module.device)
        x2 = emb2.to(pl_module.device)

        # Get unnormalized aesthetic scores
        s1 = pl_module(x1)
        s2 = pl_module(x2)
        delta_s = s1 - s2
        delta_s

        # Get y values
        y_hat = torch.sigmoid(delta_s)
        y_hat = y_hat.detach().cpu() # probability of x1 preferred x2  0.0-1.0
        y = label.detach().cpu().reshape(-1, 1)
        # If y is 0.0 or 1.0, change to 0.01 or 0.99
        y[y == 0.0] = 0.05
        y[y == 1.0] = 0.95
        # Log the images - download url
         # Get image pairs and ranks
        table = wandb.Table(columns=['img1', 'a_aes_def', 'img2', 'b_aes_def', 'pred', 'y', 'bce_loss'])
        for i in range(len(imgs1)):
            img1 = wandb.Image(imgs1[i], caption=caption1[i])
            img2 = wandb.Image(imgs2[i], caption=caption2[i])
            log_y = y[i][0]
            # access ith element of y_hat (8,1)
            log_y_hat = y_hat[i][0]
            log_loss_val = F.binary_cross_entropy(log_y_hat, log_y)
            table.add_data(img1, a_aes_def[i], img2, b_aes_def[i], log_y_hat, log_y, log_loss_val)

        # Log the table
        trainer.logger.experiment.log({
            "predictions": table
        })

# Folder which aggregates the results from running clip-retrieval on 1 or more datasets
def train(config=None):
        with wandb.init(config=config):
            lr = config["lr"] if config else wandb.config.lr
            max_epochs = config["max_epochs"] if config else wandb.config.max_epochs
            accumulate_grad_batches = config["accumulate_grad_batches"] if config else wandb.config.accumulate_grad_batches
            gradient_clip_val = config["gradient_clip_val"] if config else wandb.config.gradient_clip_val
            # Dataset
            tsv_embed_folders = ["./data/outputs/toloka/"]
            roots = {"reddit": None, "tsv": tsv_embed_folders}
            # Rank Dataset
            dataset = RankDataModule(roots=roots, batch_size=4)
            dataset.prepare_data()
            dataset.setup()
            samples = next(iter(dataset.val_dataloader()))
            embedding_size = samples['emb1'].shape[-1]
            # Aesthetic Model
            model = AestheticScoreMLP0(embedding_size, lr=lr)
            # Weights from improved-aesthetic-scorer
            s = torch.load("./model_weights/sac+logos+ava1-l14-linearMSE.pth")
            model.load_state_dict(s, strict=False)
            # Trainer for model + dataset
            wandb_logger = WandbLogger(log_model=True)
            trainer = pl.Trainer(
                logger=wandb_logger,    # W&B integration
                log_every_n_steps=10,   # set the logging frequency
                gpus=[3],               # Select GPUs
                max_epochs=max_epochs,      # number of epochs
                deterministic=True,     # keep it deterministic
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                callbacks=[ImagePredictionLogger(samples)],
                auto_scale_batch_size='binsearch',
            )
            trainer.tune(model, datamodule=dataset) # Find the batch size
            trainer.fit(model, dataset)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = f"./model_weights/{timestamp}.pth"
            torch.save(model.state_dict(), save_path)
            # Now we distill the model to a single linear layer and adjust the output to a Cumulative Distribution Function
            test()

import numpy as np
def test(model):
    def distill_model(model):
        def get_bias(model):
            zero = torch.zeros(model.input_size)
            with torch.no_grad():
               bias = model(zero)
            return bias
        def get_weights(model):
            one_hot = torch.eye(model.input_size)
            with torch.no_grad():
                weights = model(one_hot) - bias
            return weights
        bias = get_bias(model)
        weights = get_weights(model)
        distilled_model = lambda x: torch.matmul(x, weights) + bias
        return distilled_model

    def make_cdf(model, embs):    
        unnormalized_scores = model(embs)
        counts, edges = np.histogram(unnormalized_scores)  # check this is normalized to sum to 1
        probs = counts / np.sum(counts)
        ecdf = np.cumsum(probs)

        def run_cdf(unnormalized_score):
            change_point =  np.diff(np.int(unnormalized_score < edges))
            return np.dot(ecdf, change_point)
        return run_cdf
    
    # Get embeddings
    embeddings = ...
    # Distill model
    distilled_model = distill_model(model)
    # Make CDF
    run_cdf = make_cdf(distilled_model, embeddings)
    run_cdf(distilled_model(get_embedding(image)))
    


# Hyperparameter sweep - https://www.paepper.com/blog/posts/hyperparameter-tuning-on-numerai-data-with-pytorch-lightning-and-wandb/
def sweep():
    sweep_config = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'train/loss'
            },
        'parameters': {
            'max_epochs': {'values': [20, 40]},
            'lr': {'max': 0.001, 'min': 0.000001},
            'accumulate_grad_batches': {'values': [1, 2, 4, 8, 16, 32, 64, 128]},
            'gradient_clip_val': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='aesthetic-rankings2')
    wandb.agent(sweep_id, train, count=100)

config = {
    "lr": 0.0001,
    "max_epochs": 20,
    "accumulate_grad_batches": 7,
    "gradient_clip_val": 0.5
}
train(config)
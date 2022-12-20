import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from pytorch_lightning import LightningDataModule

class RedditRankDataset(Dataset):
    """Paired Embeddings Based On Reddit Upvote Tallys in a Subreddit"""
    def __init__(self, root="./path/to/clip-retrival/embeddings/"):
        super(RedditRankDataset, self).__init__()
        self.root = root
        # Load embeddings and scores for each root
        self.embeddings = self.loadEmbeddings(root)
        self.metadata = self.loadMetadata(root)
        # Check if embeddings and scores are valid
        self.valid = True if self.embeddings is not None and self.metadata is not None and len(self.embeddings) == len(self.metadata) else False

    def loadEmbeddings(self, root):
        embedding_np = os.path.join(root, 'img_emb/img_emb_0.npy')
        if not os.path.exists(embedding_np):
            return None
        x = np.load(embedding_np)
        # Convert to torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        return x
    
    def loadMetadata(self, root):
        metadata_parquet = os.path.join(root, 'metadata/metadata_0.parquet')
        if not os.path.exists(metadata_parquet):
            return None
        y = pd.read_parquet(metadata_parquet)
        return y

    def __getitem__(self, index):
        # Get two random indices
        idx1 = np.random.randint(0, len(self.embeddings))
        idx2 = np.random.randint(0, len(self.embeddings))
        # Get the embeddings
        x1 = self.embeddings[idx1]
        x2 = self.embeddings[idx2]
        # Get the metadata
        y1 = self.metadata.iloc[idx1]
        y2 = self.metadata.iloc[idx2]
        # Get the score
        score1 = y1['score']
        score2 = y2['score']
        # Get the url
        url1 = y1['url']
        url2 = y2['url']
        # Get the caption
        caption1 = y1['caption']
        caption2 = y2['caption']
        # Create the label
        label = 1 if score1 > score2 else 0
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return {'emb1': x1, 'emb2': x2, 'label': label, 'url1': url1, 'url2': url2, 'caption1': caption1, 'caption2': caption2}
    
    def __len__(self):
        return len(self.embeddings)

class MultiRankDataset(Dataset):
    """Group of Paired Embeddings with a label of 1 if the first is higher ranked than the second, 0 otherwise"""
    def __init__(self, roots=["./path/to/clip-retrival/embeddings/"]):
        super(MultiRankDataset, self).__init__()
        self.roots = roots
        # Load embeddings and scores for each root
        self.datasets = [RedditRankDataset(root) for root in roots]
        # Remove datasets where valid is False
        self.datasets = [dataset for dataset in self.datasets if dataset.valid]
        print(f"Loaded {len(self.datasets)} datasets")

    def __getitem__(self, index):
        # Get a random root
        root = np.random.randint(0, len(self.datasets))
        # Get the dataset
        dataset = self.datasets[root]
        # Get the item
        sample = dataset[index]
        return sample

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

class RankDataModule(LightningDataModule):
    def __init__(self, root="./path/to/clip-retrival/embeddings/", batch_size=8, num_workers=8):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Get top level folders
            self.roots = [os.path.join(self.root, f) for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
            self.dataset = MultiRankDataset(roots=self.roots)
            # Error Below: Sum of input lengths does not equal the length of the input dataset!
            train_len = int(len(self.dataset)*0.9)
            val_len = len(self.dataset) - train_len
            self.train_set, self.val_set = random_split(self.dataset, [train_len, val_len])
        if stage == 'test' or stage is None:
            # We could load random samples from LAION here
            pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

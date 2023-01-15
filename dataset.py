import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from pytorch_lightning import LightningDataModule
import requests
from PIL import Image
from tqdm import tqdm

class PairRankPoolDataset(Dataset):
    """Paired Embeddings Based On Pairwise Rankings of Images"""
    def __init__(self, root="./path/to/toloka"):
        super(PairRankPoolDataset, self).__init__()
        self.root = root
        # Load embeddings and scores for each root
        self.embeddings = self.loadEmbeddings(root)
        self.pair_metadata = self.loadPairMetadata(root)
        # self.emb_metadata = self.loadMetadata(os.path.join(root, 'embeddings'))
        # self.url_metadata = self.loadMetadata(os.path.join(root, 'wds'))
        print(f"Loaded {len(self.embeddings)} embeddings and {len(self.pair_metadata)} pairs")
        self.valid = True if self.embeddings is not None and self.pair_metadata is not None else False

    def loadEmbeddings(self, root):
        embedding_np = os.path.join(root, 'embeddings/img_emb/img_emb_0.npy')
        if not os.path.exists(embedding_np):
            return None
        x = np.load(embedding_np)
        # Convert to torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def loadPairMetadata(self, root):
        agreement_csv = os.path.join(root, 'agreement_results.csv')
        if not os.path.exists(agreement_csv):
            return None
        y = pd.read_csv(agreement_csv)
        return y

    def loadMetadata(self, dir):
        # Load all the parquet files in the dir and pd.concat them
        df = None
        # Get all files in a directory recursively
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for file in filenames:
                if file.endswith('.parquet'):
                    # Load parquet file
                    parquet_path = os.path.join(dirpath, file)
                    parquet_df = pd.read_parquet(parquet_path)
                    # Add to df
                    if df is None:
                        df = parquet_df
                    else:
                        df = pd.concat([df, parquet_df])
        return df

    def __getitem__(self, index):
        # Get random value
        image_a_index = np.random.randint(0, len(self.pair_metadata))
        # Find the entry
        pair = self.pair_metadata.iloc[image_a_index]
        # Get the indices of the embeddings
        image_a_index = pair['image_a_emb_idx']
        image_b_index = pair['image_b_emb_idx']
        # Get the embeddings
        x1 = self.embeddings[image_a_index]
        x2 = self.embeddings[image_b_index]
        # urls from first value in index
        url1 = pair['Unnamed: 0']
        url2 = pair['Unnamed: 1']
        # Default aesthetic scores
        image_a_aes_default = pair['image_a_pred']
        image_b_aes_default = pair['image_b_pred']
        # Convert all values to tensors
        label = torch.tensor(pair['agreement'], dtype=torch.float32)
        return {'emb1': x1, 'emb2': x2, 'label': label, 'url1': url1, 'url2': url2, 'caption1': 'toloka_a', 'caption2': 'toloka_b', 'image_a_aes_default': image_a_aes_default, 'image_b_aes_default': image_b_aes_default}

    def __len__(self):
        return len(self.pair_metadata)

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
        # Get the caption
        caption1 = y1['caption']
        caption2 = y2['caption']
        # Create the label
        label = 1 if score1 > score2 else 0
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return {'emb1': x1, 'emb2': x2, 'label': label, 
                'url1': url1, 'url2': url2, 
                'caption1': caption1, 'caption2': caption2
                }
    
    def __len__(self):
        return len(self.embeddings)

class MultiRankDataset(Dataset):
    """Group of Paired Embeddings with a label of 1 if the first is higher ranked than the second, 0 otherwise"""
    def __init__(self, roots={"reddit": [], "tsv": []}):
        super(MultiRankDataset, self).__init__()
        self.roots = roots
        # Load embeddings and scores for each root
        reddit_datasets = [RedditRankDataset(root) for root in roots["reddit"]] if roots["reddit"] is not None else []
        tsv_datasets = [PairRankPoolDataset(root) for root in roots["tsv"]] if roots["tsv"] is not None else []
        # Combine PyTorch datasets
        self.datasets = reddit_datasets + tsv_datasets
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
    def __init__(self, roots={"reddit": "./path/to/clip-retrival/embeddings/", "tsv": "./path/to/clip-retrieval/embeddings"}, batch_size=8, num_workers=8):
        super().__init__()
        self.reddit_root = roots["reddit"] if "reddit" in roots else None
        self.tsv_roots = roots["tsv"] if "tsv" in roots else None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def getRedditRoots(self):
        if self.reddit_root is None:
            return []
        # Get top level folders
        self.roots = [os.path.join(self.root, f) for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        # Keep only certain datasets
        keep_roots = ["earth", "sea", "beach"]
        exclude_roots = ["earthling"]
        # Not case sensitive
        roots = [root for root in self.roots if any([keep_root.lower() in root.lower() for keep_root in keep_roots])]
        reddit_roots = [root for root in self.roots if not any([exclude_root.lower() in root.lower() for exclude_root in exclude_roots])]
        return roots

    def addTSVDataset(self):
        # Load the LAION dataset
        dataset = PairRankPoolDataset(self.root)
        return dataset

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.reddit_roots = self.getRedditRoots()
            # Combine roots
            self.roots = {"reddit": self.reddit_roots, "tsv": self.tsv_roots}
            self.dataset = MultiRankDataset(roots=self.roots)
            # Create dataset splits
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

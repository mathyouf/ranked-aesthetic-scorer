from pairwise import AestheticScoreMLP0
from dataset import RankDataModule

#TODO: test this yo! thx charles
# We have to make a Cumulative distribution function (CDF) from the unnormalized scores
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

def getDataset():
	dataset = RankDataModule(roots={"reddit": None, "tsv": ["./data/outputs/toloka/"]}, batch_size=4)
	dataset.prepare_data()
	dataset.setup()

dataset = getDataset()
samples = next(iter(dataset.val_dataloader()))

embedding_size = samples['emb1'].shape[-1]
model = AestheticScoreMLP0(embedding_size, lr=0.001)
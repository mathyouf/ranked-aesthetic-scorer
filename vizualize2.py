"""
This is an example on how to use embedding reader to do an inference over a dataset
of clip vit-l/14 embeddings to predict tags for each example, representing the aesthetic value of images
"""

from builtins import ValueError
from embedding_reader import EmbeddingReader
import fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import fsspec
import math
import pandas as pd
import pyarrow.parquet as pq


dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

import mmh3

def compute_hash(image_path):
    total = (image_path[0]).encode("utf-8")
    return mmh3.hash64(total)[0]


def createAestehticsEmbeddings(
    embedding_folder="/home/matt/Desktop/ranked-aesthetic-scorer/data/outputs/toloka/embeddings/img_emb",
    metadata_folder="/home/matt/Desktop/ranked-aesthetic-scorer/data/outputs/toloka/embeddings/metadata",
    output_folder="/home/matt/Desktop/ranked-aesthetic-scorer/data/outputs/toloka/embeddings/aesthetics",
    batch_size=10**6,
    end=None,
):
    """main function"""
    reader = EmbeddingReader(
        embedding_folder, metadata_folder=metadata_folder, file_format="parquet_npy", meta_columns=["image_path"]
    )

    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    model = get_aesthetic_model()

    # convert numpy int64 (reader.count) value as integer
    total = int(reader.count)
    min_batch = math.ceil(total // batch_size)
    batch_count = min_batch if min_batch > 0 else 1
    padding = int(math.log10(batch_count)) + 1

    for i, (embeddings, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        with torch.no_grad():
            predictions = model(torch.tensor(embeddings)).cpu().numpy()
        padded_id = str(i).zfill(padding)
        output_file_path = os.path.join(relative_output_path, padded_id + ".parquet")
        df = pd.DataFrame(predictions, columns=["prediction"])
        df["hash"] = [compute_hash(x) for x in zip(ids["image_path"])]
        df["image_path"] = ids["image_path"]
        with fs.open(output_file_path, "wb") as f:
            df.to_parquet(f)

if __name__ == "__main__":
    fire.Fire(main)

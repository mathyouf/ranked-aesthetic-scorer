import os
import json
import pandas as pd
from tqdm import tqdm

def URS_JSON2CSV(json_path, output_dir="data/URS_CSV", filter_nsfw=True, filter_low_upvote_ratio=0.95):
    # Prepare data created by Universal Reddit Scraper (https://github.com/JosephLai241/URS) & Turn it into a dataframe of image and upvote pairs
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        # exist_ok=True prevents error if directory already exists
        os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Create dataframe
    df = pd.DataFrame(data.get('data'))
    # Keep url, score, num_comments, title, nsfw, and upvote_ratio
    df = df[['url', 'score', 'num_comments', 'title', 'nsfw', 'upvote_ratio']]
    # Filter out nsfw images
    if filter_nsfw:
        # print number to be filtered out
        print('Number of nsfw images: ' + str(len(df[df['nsfw'] == True])))
        df = df[df['nsfw'] == False]
    # print number to be filtered out
    print('Number of upvote ratios below ' + str(filter_low_upvote_ratio) + ': ' + str(len(df[df['upvote_ratio'] < filter_low_upvote_ratio])))
    # Filter out low upvote ratio images
    df = df[df['upvote_ratio'] > filter_low_upvote_ratio]    
    # Sort by score
    df = df.sort_values(by='score', ascending=False)
    # Reset index
    df = df.reset_index(drop=True)
    # Parquet name
    nsfw_tag = f'_filter_nsfw{filter_nsfw}'
    upvote_tag = '_upvote_ratio' + str(filter_low_upvote_ratio)
    subreddit = json_path.split('/')[-1].split('.')[0].split('-')[0]
    filename = subreddit + nsfw_tag + upvote_tag + "_img_count" + str(len(df))
    # Save as csv
    csv_name = filename + '.csv'
    csv_path = os.path.join(output_dir, csv_name)
    print('CSV name: ' + csv_name)
    df.to_csv(csv_path)
    return csv_path

def getJSONPaths(scrapes_subreddits_folder):
    # Get paths to all json files in a folder
    json_paths = []
    for root, dirs, files in os.walk(scrapes_subreddits_folder):
        for file in files:
            if file.endswith(".json"):
                json_paths.append(os.path.join(root, file))
    return json_paths

scrapes_subreddits_folder = "/home/matt/Desktop/scrapes/12-15-2022/subreddits"

output_dir="data/URS_CSV"

json_paths = getJSONPaths(scrapes_subreddits_folder)

if not os.path.exists(output_dir):
    csv_names = [URS_JSON2CSV(json_path, output_dir=output_dir) for json_path in json_paths]
else:
    # Get all csv files in output_dir
    csv_names = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')]

# https://github.com/rom1504/img2dataset
output_dirs = []
for csv_name in tqdm(csv_names):
    wds_output_dir = "data/webdataset/" + csv_name.split('.')[0]
    output_dirs.append(wds_output_dir)
    if not os.path.exists(wds_output_dir):
        # Only create if it doesn't exist
        os.system(f"img2dataset --url_list={csv_name} --output_folder={wds_output_dir} --output_format=webdataset --input_format=csv --url_col=url --caption_col=title --save_additional_columns=\"['score','num_comments','upvote_ratio','nsfw']\"")

# https://github.com/rom1504/clip-retrieval
clip_model = "ViT-L/14"
root_emb_dir = f"data/embeddings_{clip_model.replace('/', '_')}/"
if not os.path.exists(root_emb_dir):
    os.system(f"mkdir {root_emb_dir}")
for output_dir in tqdm(output_dirs):
    output_dir_tar = output_dir + "/00000.tar"
    ds_name = output_dir.split('/')[-1]
    embed_dir = root_emb_dir + ds_name
    # Create folder for embeddings
    if not os.path.exists(embed_dir):
        os.system(f"mkdir {embed_dir}")
    os.system(f"clip-retrieval inference --input_dataset={output_dir_tar} --clip_model={clip_model} --enable_wandb=False --output_folder={embed_dir} --input_format=webdataset --enable_metadata=True")

## Converts Toloka data to a format that can be used by the RankDataset class - Computes CLIP embeddings ##

# The best way to do this is to:
#  1. Convert this to an img2dataset dataset
#  2. Use the clip-retrieval script to compute the embeddings for the images in the toloka dataset
#  3. Use the embeddings to create a new dataset with the pairs

from utils import prepareFolders, uniqueValues, filterCol
import os
from tqdm import tqdm

def filterBadAssignments(df, filter_col='ASSIGNMENT:status'):
	statuses = uniqueValues(df, col_name=filter_col)
	# ASSIGNMENT:status == APPROVED
	APPROVED = [statuses[0]]
	df = filterCol(df, col=filter_col, vals=APPROVED)
	return df

def filterBadInputs(df, filter_col='OUTPUT:result'):
	# OUTPUT:result == image_a or image_b
	input_names = [col.split('INPUT:')[-1] for col in df.columns if col.startswith('INPUT:')]
	result_values = uniqueValues(df, col_name=filter_col)
	df = filterCol(df, col=filter_col, vals=input_names)
	return df

def printIncorrect(workers):
	count_thresholds = [1, 5, 10, 100, 500, 1000]
	for i, count_threshold in enumerate(count_thresholds):
		if i < len(count_thresholds) - 1:
			number_of_workers = len([worker for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]])
			print(f"	between {count_thresholds[i]} and {count_thresholds[i+1]} assignments: {number_of_workers} workers")
			workers_temp = [worker for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]]
			for worker in workers_temp[:10]:
				print(f"Count: {worker['count']}, Incorrect: {worker['incorrect']}, Percent: {worker['incorrect_percent']}%")

def getWorkerInfo(df):
	# Get outputs which have a GOLDEN:result value
	golden = df[df['GOLDEN:result'].notnull()]
	print('Number of golden outputs:', len(golden))
	# Get rows where OUTPUT:result != GOLDEN:result when GOLDEN:result is not None
	incorrect = golden[golden['OUTPUT:result'] != golden['GOLDEN:result']].sort_values(by=['GOLDEN:result'])
	correct = golden[golden['OUTPUT:result'] == golden['GOLDEN:result']]
	print('Number of incorrect golden outputs:', len(incorrect))
	# Sorted list of workers with the most assignments
	workers = [{'worker_id': worker_id, 'count': count} for worker_id, count in zip(df['ASSIGNMENT:worker_id'].value_counts().index.tolist(), df['ASSIGNMENT:worker_id'].value_counts().tolist())]
	# Add the number of incorrect golden outputs to each worker
	for worker in workers:
		worker['incorrect'] = len(incorrect[incorrect['ASSIGNMENT:worker_id'] == worker['worker_id']])
		worker['correct'] = len(correct[correct['ASSIGNMENT:worker_id'] == worker['worker_id']])
		# Percentage truncated to 0 decimal places
		total_golden = worker['correct']+worker['incorrect']
		if total_golden > 0:
			worker['incorrect_percent'] = int(worker['incorrect'] / (worker['correct']+worker['incorrect']) * 100)
		else:
			worker['incorrect_percent'] = -1

	# Sort workers by incorrect_percent
	workers = sorted(workers, key=lambda worker: worker['incorrect_percent'], reverse=True)
	return workers, incorrect

def filterBadWorkers(df):
	workers, incorrect = getWorkerInfo(df)
	# Filter out workers with more than 20% incorrect golden outputs
	workers_to_remove = [worker['worker_id'] for worker in workers if worker['incorrect_percent'] > 20]
	workers_to_keep = [worker for worker in workers if worker['worker_id'] not in workers_to_remove]
	before = len(df)
	df = df[~df['ASSIGNMENT:worker_id'].isin(workers_to_remove)]
	print('Removed ', len(workers_to_remove), ' workers and ', before - len(df), ' rows')
	visualizeHTML(df, workers_to_keep, incorrect)
	return df

def visualizeHTML(df, workers, incorrect):
	# Create header
	html = '<html><head><style>table, th, td {border: 1px solid black;}</style></head><body>'
	# Visualize worker data into buckets
	html += f'<h1>Workers ({len(workers)})</h1>'
	count_thresholds = [1, 10, 100, 500, 1000]
	for i, count_threshold in enumerate(count_thresholds):
		if i < len(count_thresholds) - 1:
			html += f'<h2>Between {count_thresholds[i]} and {count_thresholds[i+1]} assignments</h2>'
			html += '<table>'
			html += '<tr><th>Worker ID</th><th>Count</th><th>Incorrect</th><th>Incorrect Percent</th></tr>'
			workers_temp = [worker for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]]
			for worker in workers_temp[:40]:
				html += '<tr>'
				html += f'<td>{worker["worker_id"]}</td>'
				html += f'<td>{worker["count"]}</td>'
				html += f'<td>{worker["incorrect"]}</td>'
				html += f'<td>{worker["incorrect_percent"]}%</td>'
				html += '</tr>'
			html += '</table>'
	# Create table where each row in incorrect has an img tag for `image_a` and `image_b` and the `OUTPUT:result` value and the `GOLDEN:result` value
	# Title
	html += f'<h1>Incorrect Golden Outputs ({len(incorrect)})</h1>'
	html += '<table>'
	for i, row in incorrect.iterrows():
		url_a = row['INPUT:image_a']
		url_b = row['INPUT:image_b']
		given_result = row['OUTPUT:result']
		golden_result = row['GOLDEN:result']	
		html += '<tr>'
		html += f'<td><img src="{url_a}" width="256"></td>'
		html += f'<td><img src="{url_b}" width="256"></td>'
		html += f'<td>{given_result}</td>'
		html += f'<td>{golden_result}</td>'
		# Add button to mark entry
		html += '</tr>'
		# Add button to mark entry
	html += '</table>'
	html += '</body></html>'
	# Save html
	html_path = os.path.join(session_dir, 'workers.html')
	with open(html_path, 'w') as f:
		f.write(html)

def uniqueComparisons(df, unique_urls):
	unique_urls_path = os.path.join(session_dir, 'unique_urls.csv')
	# If unique_urls.csv exists, load it
	if os.path.exists(unique_urls_path):
		unique_urls = pd.read_csv(unique_urls_path)
	# Else, create it
	else:
		# Add two columns to unique_urls, one for better_indices and one for worse_indices
		unique_urls['better_indices'] = [[] for i in range(len(unique_urls))]
		unique_urls['worse_indices'] = [[] for i in range(len(unique_urls))]
		# For each comparision in df using tqdm
		for i, row in tqdm(df.iterrows(), total=len(df)):
			# Get the indices of the two urls in unique_urls
			url_a = row['INPUT:image_a']
			url_b = row['INPUT:image_b']
			# Get the row index value of the url in unique_urls
			url_a_index = unique_urls.loc[unique_urls['url'] == url_a].index[0]
			url_b_index = unique_urls.loc[unique_urls['url'] == url_b].index[0]
			# Get the result of the comparision
			result = row['OUTPUT:result']
			# Add the indicies to the better_indices and worse_indices columns
			if result == 'image_a':
				unique_urls.at[url_a_index, 'worse_indices'].append(url_b_index)
				unique_urls.at[url_b_index, 'better_indices'].append(url_a_index)
			elif result == 'image_b':
				unique_urls.at[url_a_index, 'better_indices'].append(url_b_index)
				unique_urls.at[url_b_index, 'worse_indices'].append(url_a_index)
		# Save df with preserved indicies
		unique_urls.to_csv(unique_urls_path, index=False)
	# Iterate through unique_urls and find where the better_indices and worse_indices are the same
	# Create dataframe called disagreeing_urls with two columns, image_a and image_b
	disgareement_path = os.path.join(session_dir, 'disagreements.csv')
	# If disagreements.csv exists, load it
	if os.path.exists(disgareement_path):
		disagreeing_urls = pd.read_csv(disgareement_path)
	# Else, create it
	else:
		disagreeing_urls = []
		for i, row in tqdm(unique_urls.iterrows(), total=len(unique_urls)):
			entry = {"index": i, "url": row['url']}
			# Get the better_indices and worse_indices
			better_indices = row['better_indices']
			if better_indices == '[]':
				continue
			print(better_indices)
			better_indices_parsed = [int(index) for index in better_indices[1:-1].split(',')]
			worse_indices = row['worse_indices']
			if worse_indices == '[]':
				continue
			worse_indices_parsed = [int(index) for index in worse_indices[1:-1].split(',')]
			# For each unique_url, get the number of times another url is in the better_indices and worse_indices, and add it to a dictionary as a percentage
			for index in better_indices_parsed:
				if index in worse_indices_parsed:
					entry['disagreement'] = 1
					disagreeing_urls.append(entry)
					break

		# Save list of disagreeing urls by convrting to a dataframe and saving as csv
		disagreeing_urls.to_csv(os.path.join(session_dir, 'disagreeing_urls.csv'), index=False)
	# Return unique_urls
	return unique_urls

def getUniqueURLS(df):
	# Create new dataframe with a url column
	img_a_df = pd.DataFrame(columns=['url'])
	img_a_df['url'] = df['INPUT:image_a']
	# Add new rows for image_b
	img_b_df = pd.DataFrame(columns=['url'])
	img_b_df['url'] = df['INPUT:image_b']
	# Concatenate the two dataframes
	wds_df = pd.concat([img_a_df, img_b_df])
	# Get unique url count
	unique_urls = wds_df.drop_duplicates().reset_index()
	print(f'Unique urls: {len(unique_urls)}')
	return unique_urls

def dfToHTML(df, session_dir, html_name='low_agreement.html'):
	# Convert index entries to img_a and img_b
	df['img_url_a'] = df.index.map(lambda x: x[0])
	df['img_url_b'] = df.index.map(lambda x: x[1])
	# Form the column orderings
	df = df[['img_url_a', 'image_a_pred', 'image_a', 'image_b', 'image_b_pred', 'img_url_b', 'agreement']]
	# Drop the index
	df = df.reset_index(drop=True)
	# pick two good neutral opposite colors
	image_a_color = "blue"
	image_b_color = "gold"
	# Map img_url_a and img_url_b to img html
	# Put green border around image_a and red border around image_b
	df['img_url_a'] = df['img_url_a'].map(lambda x: '''<img src="''' + x + f'''" width="256" style="max-height:256px; border: 5px solid {image_a_color}">''')
	df['img_url_b'] = df['img_url_b'].map(lambda x: '''<img src="''' + x + f'''" width="256" style="max-height:256px; border: 5px solid {image_b_color}">''')

	# Create new column with the difference between image_a_pred and image_b_pred
	df['aesthetic_difference'] = df['image_a_pred'] - df['image_b_pred']
	# Round to nearest 2 decimals
	df['aesthetic_difference'] = df['aesthetic_difference'].map(lambda x: round(x, 2))
	# If its positive, make it green
	df['aesthetic_difference'] = df['aesthetic_difference'].map(lambda x: f'''<span style="color:{image_a_color}">????????{x}</span>''' if x > 0 else f'''<span style="color:{image_b_color}">????????{x}</span>''')

	# Round agreement to nearest 2 decimals
	df['agreement'] = df['agreement'].map(lambda x: round(x*100, 2))
	# If agreement is >0.5, make it green
	df['agreement_html'] = df['agreement'].map(lambda x: f'''<span style="color:{image_a_color}">???????????{x}%</span>''' if x > 50 else f'''<span style="color:{image_b_color}">???????????{x}%</span>''')

	# If image_a_pred - image_b_pred is positive, then agreement should be >0.5
	df_l = df[(df['image_a_pred'] - df['image_b_pred']) > 1.0] # A is aesthetically better than B by more than 0.5
	df_l = df_l[df_l['agreement'] < 0.3] # A is rated better than B less than 0.4 of the time
	df_r = df[(df['image_a_pred'] - df['image_b_pred']) < -1.0] # B is aesthetically better than A by more than 0.5
	df_r = df_r[df_r['agreement'] > 0.7] # B is rated better than A more than 0.6 of the time
	# Concatenate the two dataframes
	df_aes_agree_mismatch = pd.concat([df_l, df_r])
	# Sort by aesthetic difference
	df_aes_agree_mismatch = df_aes_agree_mismatch.sort_values(by=['aesthetic_difference'], ascending=False)

	# Form the column orderings
	df_aes_agree_mismatch = df_aes_agree_mismatch[['image_a_pred', 'image_a', 'img_url_a', 'aesthetic_difference', 'agreement_html', 'img_url_b', 'image_b', 'image_b_pred']]

	# Create html file of the dataset
	df_html = df_aes_agree_mismatch.to_html(render_links=True,escape=False)
	with open(os.path.join(session_dir, html_name), 'w') as f:
		f.write(df_html)

def filterTrait(df, col='agreement', n=0.7, s='<', name='low_agreement'):
	# Low Agreement Pairs
	filter_df = df[df[col] < n] if s == '<' else df[df[col] > n]
	filter_df.sort_values(by=col, ascending=True)
	print(filter_df.head(10))
	dfToHTML(filter_df, session_dir, name+'.html')
	filter_df.to_csv(os.path.join(session_dir, name+'.csv'))
	return filter_df

import pandas as pd
def tsvTocsv(tsv_path, session_dir, csv_name='wds.csv'):
	df = pd.read_csv(tsv_path, sep='\t')
	unique_urls = getUniqueURLS(df)
	csv_name = os.path.join(session_dir, csv_name)
	unique_urls.to_csv(csv_name, index=False)
	return csv_name

def loadMetadata(dir):
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

def makeWebDataset(csv_name, session_dir, sub_folder='wds'):
	# Create webdataset dataset
	wds_output_dir = os.path.join(session_dir, sub_folder)
	if not os.path.exists(wds_output_dir):
		os.makedirs(wds_output_dir)
		print(f'Creating WebDataset of images at {wds_output_dir} using {csv_name}')
		os.system(f"img2dataset --url_list={csv_name} --output_folder={wds_output_dir} --output_format=webdataset --input_format=csv --url_col=url")
	return wds_output_dir

def makeEmbeddings(wds_output_dir, session_dir, sub_folder='embeddings'):
	# https://github./rom1504/clip-retrieval
	clip_model = "ViT-L/14"
	root_emb_dir = os.path.join(session_dir, sub_folder)
	if not os.path.exists(root_emb_dir):
		# Get list of all .tar files
		tar_files = [f for f in os.listdir(wds_output_dir) if f.endswith('.tar')]
		# Create {000..nnn}.tar style files
		# Get largest tar file number
		max_tar_num = 0
		for tar_file in tar_files:
			tar_num = int(tar_file.split('.')[0])
			if tar_num > max_tar_num:
				max_tar_num = tar_num
		# Create string
		max_tar_num = str(max_tar_num).zfill(5)
		tar_group_string = "{00000.." + str(max_tar_num) + "}.tar"
		tar_path = f"./{wds_output_dir}/{tar_group_string}"
		# Create new tar files
		os.makedirs(root_emb_dir)
		print(f"Creating embeddings for {tar_path}")
		os.system(f"clip-retrieval inference --input_dataset={tar_path} --clip_model={clip_model}, --enable_wandb=False --enable_text=False --output_folder={root_emb_dir} --input_format=webdataset")
		# Create list of pairs with valid images and embeddings
	return root_emb_dir

from builtins import ValueError
from embedding_reader import EmbeddingReader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import fsspec
import math
import pandas as pd

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

def calcEmbeddingAesthetics(emb_dir):
	embedding_folder=os.path.join(emb_dir, "img_emb")
	metadata_folder=os.path.join(emb_dir, "metadata")
	output_folder=os.path.join(emb_dir, "aesthetics")
	batch_size=10**6
	end=None

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

def filterFailures(toloka_df, img_meta_df):
	# Find rows in img_meta_df where "status" != "success"
	failures = img_meta_df[img_meta_df['status'] != 'success']
	# Remove entries in toloka_df where "INPUT:image_a" or "INPUT:image_b" are in failures
	before = len(toloka_df)
	print(f'Before removing failed images, toloka_df has {before} rows')
	for row in tqdm(failures.iterrows(), total=len(failures)):
		url = row[1]['url']
		toloka_df = toloka_df[(toloka_df['INPUT:image_a'] != url) & (toloka_df['INPUT:image_b'] != url)]
	after = len(toloka_df)
	print(f'Removed {before - after} rows from toloka_df due to failed images')
	return toloka_df

def filterToloka(toloka_df, img_meta_df):
	toloka_df = filterBadAssignments(toloka_df) # ASSIGNMENT:status != APPROVED
	toloka_df = filterBadInputs(toloka_df) # OUTPUT:result != INPUT:image_a | INPUT:image_b
	toloka_df = filterFailures(toloka_df, img_meta_df)
	return toloka_df

def makePairDF(toloka_df, meta_df):
	url_groups = toloka_df.groupby(['INPUT:image_a', 'INPUT:image_b'])
	url_results = {}
	for name, group in tqdm(url_groups):
		a_wins = group[group['OUTPUT:result'] == 'image_a'].shape[0]
		b_wins = group[group['OUTPUT:result'] == 'image_b'].shape[0]
		meta_image_a = meta_df[meta_df['url'] == name[0]]
		meta_image_b = meta_df[meta_df['url'] == name[1]]
		# Check if both rows exist
		if meta_image_a.shape[0] == 0 or meta_image_b.shape[0] == 0:
			# This image url didn't make it V_V
			continue
		# Get which index of meta_df the meta_image_a row is at
		image_a_emb_idx = meta_image_a.index[0]
		image_b_emb_idx = meta_image_b.index[0]
		# Round to three decimal places
		a_pred = round(meta_image_a['prediction'].values[0], 2)
		b_pred = round(meta_image_b['prediction'].values[0], 2)
		# Filter urls not with embeddings
		url_results[name] = {	'image_a': a_wins, 'image_b': b_wins, 
								'image_a_emb_idx': image_a_emb_idx, 'image_b_emb_idx': image_b_emb_idx,
								'image_a_pred': a_pred, 'image_b_pred': b_pred}
	# Convert to agreeent ratio
	for key, value in url_results.items():
		total = value['image_a'] + value['image_b']
		value['agreement'] = value['image_a'] / total
	results_df = pd.DataFrame.from_dict(url_results, orient='index')
	results_df = results_df.sort_values(by='agreement', ascending=False)
	results_df.to_csv(os.path.join(session_dir, 'agreement_results.csv'))
	filterTrait(results_df, col='agreement', n=1.1, s='<', name='low_agreement') # just for visualization using html
	return results_df

def createPairs(wds_dir, emb_dir, tsv_path, session_dir, name='toloka5'):
	data_parquet_path = os.path.join(session_dir, name+'.parquet')
	meta_df_path = os.path.join(session_dir, 'meta_df.parquet')
	if not os.path.exists(data_parquet_path):
		toloka_df = pd.read_csv(tsv_path, sep='\t')
		img_meta_df = loadMetadata(wds_dir) # ["key"]
		toloka_df = filterToloka(toloka_df, img_meta_df)
		emb_meta_df = loadMetadata(os.path.join(emb_dir, 'metadata')) # ["image_path"]
		eb_aes_df = loadMetadata(os.path.join(emb_dir, 'aesthetics'))
		# remove hash column
		eb_aes_df = eb_aes_df.drop(columns=['hash'])
		# Filter img_meta_df to only include entries where "status" == "success"
		img_meta_df = img_meta_df[img_meta_df['status'] == 'success']
		# Merge img_meta_df and emb_meta_df on "key" and "image_path", remove entries where no match is found
		meta_df = pd.merge(img_meta_df, emb_meta_df, left_on='key', right_on='image_path')
		# Merge meta_df and eb_aes_df on "key" and "image_path", remove entries where no match is found
		meta_df = pd.merge(meta_df, eb_aes_df, left_on='key', right_on='image_path')
		# Drop image_path_xn and image_path_y
		meta_df = meta_df.drop(columns=['image_path_x', 'image_path_y', 'md5', 'exif'])
		meta_df.to_parquet(meta_df_path)
		toloka_df.to_parquet(data_parquet_path)
	else:
		meta_df = pd.read_parquet(meta_df_path)
		toloka_df = pd.read_parquet(data_parquet_path)
	pair_df = makePairDF(toloka_df, meta_df)
	# Add aesthetic scores to pair_df

session_dir = prepareFolders()
tsv_path = 'data/inputs/toloka/assignments_from_pool_36836296__16-12-2022.tsv'

# Create Embeddings: (762,)
csv_name = tsvTocsv(tsv_path, session_dir)
wds_dir = makeWebDataset(csv_name, session_dir)
emb_dir = makeEmbeddings(wds_dir, session_dir)
calcEmbeddingAesthetics(emb_dir)

# Create Pairs: image_a, image_b, preference (-1.0 to 1.0)
createPairs(wds_dir, emb_dir, tsv_path, session_dir)

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

def filterBadWorkers(df):
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
	print('Workers with the most incorrect golden outputs:')
	count_thresholds = [1, 5, 10, 100, 500, 1000]
	for i, count_threshold in enumerate(count_thresholds):
		if i < len(count_thresholds) - 1:
			number_of_workers = len([worker for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]])
			print(f"	between {count_thresholds[i]} and {count_thresholds[i+1]} assignments: {number_of_workers} workers")
			workers_temp = [worker for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]]
			for worker in workers_temp[:10]:
				print(f"Count: {worker['count']}, Incorrect: {worker['incorrect']}, Percent: {worker['incorrect_percent']}%")
	# Graph data
	plot_paths = graphData(df, workers, incorrect)
	# Filter out workers with more than 20% incorrect golden outputs
	workers_to_remove = [worker['worker_id'] for worker in workers if worker['incorrect_percent'] > 20]
	print('Number of workers to remove:', len(workers_to_remove))
	print('Number of rows before removing workers:', len(df))
	df = df[~df['ASSIGNMENT:worker_id'].isin(workers_to_remove)]
	print('Number of rows after removing workers:', len(df))
	workers = [worker for worker in workers if worker['worker_id'] not in workers_to_remove]
	visualizeHTML(df, workers, incorrect, plot_paths)
	return df

import matplotlib.pyplot as plt
import numpy as np
def graphData(df, workers, incorrect):
	plot_paths = []
	# Plot the distribution of the number of assignments per worker
	assignments_per_worker = df['ASSIGNMENT:worker_id'].value_counts().tolist()
	plt.hist(assignments_per_worker, bins=100)
	plt.title('Distribution of the number of assignments per worker')
	plt.xlabel('Number of assignments')
	plt.ylabel('Number of workers')
	assignments_path = 'assignments_per_worker.png'
	plot_paths.append(assignments_path)
	plt.savefig(os.path.join(session_dir, assignments_path))
	plt.clf()
	# Plot the distribution of the incorrect percentage per worker
	count_thresholds = [1, 10, 100, 500, 1000]
	for i, count_threshold in enumerate(count_thresholds):
		if i < len(count_thresholds) - 1:
			incorrect_percentages = [worker['incorrect_percent'] for worker in workers if worker['count'] >= count_thresholds[i] and worker['count'] < count_thresholds[i+1]]
			plt.hist(incorrect_percentages, bins=100)
			plt.title(f'Distribution of the incorrect percentage per worker between {count_thresholds[i]} and {count_thresholds[i+1]} assignments')
			plt.xlabel('Incorrect percentage')
			plt.ylabel('Number of workers')
			incorrect_path = f'incorrect_percentages_{count_thresholds[i]}_{count_thresholds[i+1]}.png'
			plot_paths.append(incorrect_path)
			plt.savefig(os.path.join(session_dir, incorrect_path))
			plt.clf()
	return plot_paths

def visualizeHTML(df, workers, incorrect, plot_paths):
	# Create header
	html = '<html><head><style>table, th, td {border: 1px solid black;}</style></head><body>'
	# Visualize the plot paths
	for plot_path in plot_paths:
		html += f'<img src="{plot_path}">'
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

def saveAsCSV(df):
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
	# Get the comparrisons for each unique url
	# uniquedf = uniqueComparisons(df, unique_urls)
	# Save as csv
	csv_name = os.path.join(session_dir, 'wds.csv')
	unique_urls.to_csv(csv_name, index=False)
	return csv_name

def findDisagreements(df):
	url_groups = df.groupby(['INPUT:image_a', 'INPUT:image_b'])
	url_results = {}
	for name, group in url_groups:
		a_wins = group[group['OUTPUT:result'] == 'image_a'].shape[0]
		b_wins = group[group['OUTPUT:result'] == 'image_b'].shape[0]
		url_results[name] = {'image_a': a_wins, 'image_b': b_wins}
	# Convert to ratio
	for key, value in url_results.items():
		total = value['image_a'] + value['image_b']
		value['agreement'] = value['image_a'] / total if value['image_a'] > value['image_b'] else value['image_b'] / total
	results_df = pd.DataFrame.from_dict(url_results, orient='index')
	results_df = results_df.sort_values(by='agreement', ascending=False)
	results_df.to_csv(os.path.join(session_dir, 'agreement_results.csv'))
	# Get the urls for the top 10% of disagreements
	top_10_percent = results_df[results_df['agreement'] < 0.7]
	# Convert index entries to img_a and img_b
	top_10_percent['img_url_a'] = top_10_percent.index.map(lambda x: x[0])
	top_10_percent['img_url_b'] = top_10_percent.index.map(lambda x: x[1])
	# Move img_url_a to the first column
	top_10_percent = top_10_percent[['img_url_a', 'image_a', 'image_b','img_url_b', 'agreement']]
	# Drop the index
	top_10_percent = top_10_percent.reset_index(drop=True)
	# Convert img_url_a and img_url_b to img tags using '''<img src="''' + df1['image'] + '''">''' as the map function with width of 256 px and max height of 256 px
	top_10_percent['img_url_a'] = top_10_percent['img_url_a'].map(lambda x: '''<img src="''' + x + '''" width="256" style="max-height:256px">''')
	top_10_percent['img_url_b'] = top_10_percent['img_url_b'].map(lambda x: '''<img src="''' + x + '''" width="256" style="max-height:256px">''')
	# Create html file with the top 10% of disagreements
	top_10_percent_html = top_10_percent.to_html(render_links=True,escape=False)
	with open(os.path.join(session_dir, 'top_10_percent.html'), 'w') as f:
		f.write(top_10_percent_html)
	# Print the top ten entries
	print(top_10_percent.head(10))
	top_10_percent.to_csv(os.path.join(session_dir, 'top_10_percent.csv'))

import pandas as pd
def loadTSV(root):
	# Load .tsv file
	df = pd.read_csv(root, sep='\t')
	df = filterBadAssignments(df)
	df = filterBadInputs(df)
	df = filterBadWorkers(df)
	disagreements = findDisagreements(df)
	csv_name = saveAsCSV(df)
	return csv_name, df

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

session_dir = prepareFolders()
csv_name, toloka_df = loadTSV('data/inputs/toloka/assignments_from_pool_36836296__16-12-2022.tsv')
wds_output_dir = os.path.join(session_dir, 'wds')
if not os.path.exists(wds_output_dir):
	os.makedirs(wds_output_dir)
	print(f'Creating WebDataset of images at {wds_output_dir} using {csv_name}')
	os.system(f"img2dataset --url_list={csv_name} --output_folder={wds_output_dir} --output_format=webdataset --input_format=csv --url_col=url")

# https://github./rom1504/clip-retrieval
clip_model = "ViT-L/14"
root_emb_dir = os.path.join(session_dir, 'embeddings/')
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
	# Create new tar filesw
	os.makedirs(root_emb_dir)
	print(f"Creating embeddings for {tar_path}")
	os.system(f"clip-retrieval inference --input_dataset={tar_path} --clip_model={clip_model}, --enable_wandb=False --enable_text=False --output_folder={root_emb_dir} --input_format=webdataset")
	# Create list of pairs with valid images and embeddings

data_parquet_path = os.path.join(session_dir, 'toloka.parquet')
if not os.path.exists(data_parquet_path):
	img_meta_df = loadMetadata(wds_output_dir) # ["key"]
	emb_meta_df = loadMetadata(root_emb_dir) # ["image_path"]
	def getEmbeddingIndex(image_url):
		# Find the entry in img_meta_df where "url"	== image_a
		image_row = img_meta_df[img_meta_df['url'] == image_url]
		# Check if the url was found
		if len(image_row) == 0:
			return None
		# Check if image_row status is "success"
		if image_row['status'].values[0] != 'success':
			return None
		image_key = image_row['key'].values[0]
		image_emb_row = emb_meta_df[emb_meta_df['image_path'] == image_key]
		# Check if the embedding was found
		if len(image_emb_row) == 0:
			return None
		image_emb_index = image_emb_row.index[0]
		return image_emb_index

	new_df = pd.DataFrame(columns=['image_a_idx', 'image_b_idx', 'result'])
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
	# Iterate through the Toloka dataset
	print('Iterating through Toloka dataset')
	skipped = 0
	for row in tqdm(toloka_df.iterrows(), total=len(toloka_df)):
		image_a, image_b, result = row[1]['INPUT:image_a'], row[1]['INPUT:image_b'], row[1]['OUTPUT:result']
		# Find index in embedding
		image_a_emb_idx = getEmbeddingIndex(image_a)
		image_b_emb_idx = getEmbeddingIndex(image_b)
		# Check if the url was found
		if image_a_emb_idx is None or image_b_emb_idx is None:
			skipped += 1
			break
		# Add to new_df us pd.concat
		new_df = pd.concat([new_df, pd.DataFrame([[image_a_emb_idx, image_b_emb_idx, result]], columns=['image_a_idx', 'image_b_idx', 'result'])])

	print(f'Skipped {skipped} rows, {len(new_df)} rows left')
	# Save new_df as parquet
	new_df.to_parquet(os.path.join(session_dir, 'toloka.parquet'))
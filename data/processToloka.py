## Converts Toloka data to a format that can be used by the RankDataset class - Computes CLIP embeddings ##

# The best way to do this is to:
#  1. Convert this to an img2dataset dataset
#  2. Use the clip-retrieval script to compute the embeddings for the images in the toloka dataset
#  3. Use the embeddings to create a new dataset with the pairs

from utils import prepareFolders, uniqueValues, filterCol
import os

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

def saveAsCSV(df):
	# Create new dataframe with a url column
	img_a_df = pd.DataFrame(columns=['url'])
	img_a_df['url'] = df['INPUT:image_a']
	# Add new rows for image_b
	img_b_df = pd.DataFrame(columns=['url'])
	img_b_df['url'] = df['INPUT:image_b']
	# Concatenate the two dataframes
	wds_df = pd.concat([img_a_df, img_b_df])
	# Save as csv
	csv_name = os.path.join(session_dir, 'wds.csv')
	wds_df.to_csv(csv_name, index=False)
	return csv_name

import pandas as pd
def loadTSV(root):
	# Load .tsv file
	df = pd.read_csv(root, sep='\t')
	df = filterBadAssignments(df)
	df = filterBadInputs(df)
	df = filterBadWorkers(df) # TODO
	csv_name = saveAsCSV(df)
	return csv_name

session_dir = prepareFolders()
csv_name = loadTSV('data/inputs/toloka/assignments_from_pool_36836296__16-12-2022.tsv')
wds_output_dir = os.path.join(session_dir, 'wds')
if not os.path.exists(wds_output_dir):
	os.makedirs(wds_output_dir)
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
	os.system(f"clip-retrieval inference --input_dataset={tar_path} --clip_model={clip_model} --enable_wandb=False --enable_text=False --output_folder={root_emb_dir} --input_format=webdataset")
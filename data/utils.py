import os
import datetime
def prepareFolders(outputs_dir='data/outputs', source_type='toloka', new=False):
	os.makedirs(outputs_dir, exist_ok=True)
	source_type_outputs_dir = os.path.join(outputs_dir, source_type)
	os.makedirs(source_type_outputs_dir, exist_ok=True)
	if new:
		# Recreates a new folder from scratch, good for different datasets (takes time to re-download and re-embed)
		now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		session_dir = os.path.join(source_type_outputs_dir, now)
		os.makedirs(session_dir, exist_ok=True)
		return session_dir
	else:
		return source_type_outputs_dir

def uniqueValues(df, col_name='OUTPUT:result'):
	unique_values = [str(value) for value in list(df[col_name].unique())]
	for value in unique_values:
		if value == 'nan':
			nan_count = len(df[df[col_name].isna()])
			print(f"Status: {value}, Count: {nan_count}")
		else:
			count = len(df[df[col_name] == value])
			print(f"Value: {value}, Count: {count}")
	return unique_values

def filterCol(df, col="ASSIGNMENT:status", vals=['APPROVED']):
	pre_filter_count = len(df)
	df = df[~df[col].isna()]
	df = df[df[col].isin(vals)]
	post_filter_count = len(df)
	# Describe the filter
	print(f"{col} Filter: {pre_filter_count - post_filter_count} images removed due to bad inputs \n")
	return df

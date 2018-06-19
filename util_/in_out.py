import csv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from operator import itemgetter
import os
import util_.util 
from pprint import pprint
import pandas as pd
import codecs
import ast
from sksurv.preprocessing import OneHotEncoder
import pickle
from tqdm import *

def read_csv(f, delim=',', index_col='pseudopatnummer'):
	'''opens a csv reader object'''
	# return csv.reader(open(f, 'r'), delimiter=delim)
	return pd.read_csv(f, sep=';', index_col=index_col, encoding = "ISO-8859-1") #index_col='pseudopatnummer'
	# return pd.read_csv(open(f, 'r'), sep=delim,encoding='latin-1')

def read_csv2(f, delim=','): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def write_csv(f):
	'''opens a csv writer object'''	
	return csv.writer(open(f,"w"))

def iter_to_csv(iterator, f):
	'''writes the contents of a generator to a csv file'''
	out = write_csv(f)
	for row in iterator:
		out.writerow(row)

def get_file_name(path, extension=True):
	'''returns the base name of path, with (default) or without extension'''
	# get file name
	f = os.path.basename(path)

	# remove file extension if desired
	if not extension:
		f = f[:f.rfind('.')]
	
	return f

def dict2csv(d, f):
	'''write a dictionary to csv format file'''
	out = write_csv(f)
	if len(d) == 0: 
		return
	if type(d.values()[0]) == list:
		for k, v in d.items():
			out.writerow([k] + [str(el) for el in v])
	else:
		for k, v in d.items():
			out.writerow([k, v])

def pprint_to_file(f_out, obj):
	'''performs the pretty print operation to the specified file with the specified data object'''
	with open (f_out, 'w') as out:
		pprint(obj, out)

def get_headers(row): #deze zelf toegevoegd uit util.py. want import util gaf error
	'''returns the non-capitalised and bugfixed version of the header'''
	headers = [el.lower() for el in row]
	headers[0] = headers[0].split("\xef\xbb\xbf")[1] if headers[0].startswith('\xef') else headers[0] # fix funny encoding problem
	return headers

def import_data(f, record_id, target_id, survival):
	# '''imports the data and converts it to X (input) and y (output) data vectors'''

	rows = read_csv2(f)
	headers = get_headers(next(rows))

	# save column names as headers, save indices of record and target IDs

    # try:
    #     record_col = headers.index(record_id)
    #     target_col = headers.index(target_id)
    # except:
    #     print ('The specified instance ID was not found as column name. Manually check input file for correct instance ID column.')
    #     return False, False, False

	# save and split records
	print ('  ...(loading)')
	records = [row[1:] for row in rows]
	print ('  ...(converting to matrix)')
	records = np.matrix(records)
	X = records[:,0:-1] # features
	headers = headers[1:-1]

	# output
	y = records[:,-1] # target

	if survival == False:
		y=np.squeeze(np.asarray(y.astype(np.int)))

		print ('  ...(converting data type)')

		X = X.astype(np.float64, copy=False)
		y = y.astype(np.float64, copy=False)
		index_list = None

	if survival == True:
		target_list = []

		y=np.squeeze(np.asarray(y.astype(list)))
		X = X.astype(np.float64, copy=False)

		index_list = []		
		for idx, target in tqdm(enumerate(y)):
			target = eval(target)
			tuple_target = tuple(target)
			if tuple_target[1] <= 0:
				index_list.append(idx)
				continue

			target_list.append(tuple_target)

		y = np.array(target_list, dtype=[('Status', '?'), ('Survival in days', '<f8')])

		X = np.delete(X, (index_list), axis=0)

		# print(target_list)

		print ('  ...(converting data type)')
		# X = X.astype(np.float64, copy=False)
		# print(X)
		# X = OneHotEncoder().fit_transform(X)


	return X, y, headers, index_list


def to_int(l):
	return [int(el) for el in l]

def save_results(f, titles, results, distribution_info):
	'''save algorithm results'''
	out = write_csv(f)
	out.writerow([titles[0]] + results[0].tolist()) # false pos rate for ROC
	out.writerow([titles[1]] + results[1].tolist()) # true pos rate for ROC
	out.writerow([titles[2]] + [results[2]]) # AUC value
	out.writerow([titles[3]] + ["", "Pred 0", "Pred 1"]) # confusion matrix line 1
	out.writerow(["", "Actual 0"] + results[3].tolist()[0]) # confusion matrix line 2
	out.writerow(["", "Actual 1"] + results[3].tolist()[1]) # confusion matrix line 3
	out.writerow([''])
	out.writerow(['# stroke cases', '# Instances'])
	out.writerow(distribution_info)



def save_features(f, features):
	'''writes all features to file'''
	out = write_csv(f)
	for feature in sorted(features, key=itemgetter(1), reverse=True):
		out.writerow(feature)

def save_ROC(f, curves, clear=True, random=True, title='ROC Curve'):
	if clear: plt.clf() # clear

	# make picture pretty
	plt.rc('axes', color_cycle=['r', 'g', 'b', 'y', 'c', 'm', 'k'])
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)

	# plot results
	if random: plt.plot([0, 1], [0, 1], label='Random')

	mean_x = curves[0][1]
	sum_y = np.zeros(mean_x.shape)
	sum_auc = 0

	for result in curves:
		assert(type(result[1]) == np.ndarray)
		assert(type(result[2]) == np.ndarray)

		sum_y = sum_y + result[2]
		sum_auc = sum_auc + float(result[3])

		plt.plot(result[1], result[2],
			# label=result[0] + ' (AUC = %0.2f)' % result[3], lw=1)
			label=result[0].split('.csv')[0] + ' (%0.2f)' % result[3], lw=1)

	# plt.plot(mean_x, (sum_y/float(len(curves))),
	# 	label='Mean (%0.2f)' % (sum_auc/float(len(curves))), lw=3)

	# add legend
	plt.legend(loc="lower right")

	# save to file
	plt.savefig(f)

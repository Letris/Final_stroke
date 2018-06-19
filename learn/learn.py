import util_.util as util
import util_.in_out as in_out
import learn.algorithms as ML
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import simplejson
import pandas as pd
from tqdm import *

def execute(in_dir, out_dir, record_id, target_id, algorithms, feature_selection, separate_testset, in_dir_test, survival, oversampling, undersampling, aggregation):
	'''executes the learning task on the data in in_dir with the algorithms in algorithms.
		The results are written to out_dir and subdirectories,
	    and the record_ and target_ids are used to differentiate attributes and non-attributes'''
	print ('### executing learning algorithms on... ###')
	
	# get the files
	files = util.list_dir_csv(in_dir)

	# stop if no files found
	if not files:
		print ('No appropriate csv files found. Select an input directory with appropriate files')
		return

	if separate_testset:
		files_test = util.list_dir_csv(in_dir_test)
	else:
		files_test = files

	# create directory
	util.make_dir(out_dir)

	# execute each algorithm
	for alg in algorithms:
		print ('...{}'.format(alg))
	
		util.make_dir(out_dir+'/'+alg+'/')
		results_list = []	
		if separate_testset:
			results_list2 = []
			util.make_dir(out_dir+'/'+alg+'_test/')

		# list which will contain the results
	
		# run algorithm alg for each file f
		for f, f_test in zip(files,files_test):
			fname = in_out.get_file_name(f, extension=False)
			print (' ...{}'.format(fname))
	
			# get data, split in features/target. If invalid stuff happened --> exit
			X, y, headers, target_list = in_out.import_data(f, record_id, target_id, survival) # assumption: first column is patientnumber and is pruned, last is target
			if type(X) == bool: return
		
			if aggregation == True:
				X, headers = aggregations(f, target_list, survival)

			print ('  ...instances: {}, attributes: {}'.format(X.shape[0], X.shape[1]))

			model, best_features, results = execute_with_algorithm(alg, X, y, fname, headers, out_dir+'/'+alg+'/', record_id, target_id, feature_selection, oversampling, survival, undersampling, aggregation)
			results_list.append(results)

			if separate_testset:
				X, y, headers = in_out.import_data(f_test, record_id, target_id) # assumption: first column is patientnumber and is pruned, last is target
				if type(X) == bool: return
				
				print ('  ...instances: {}, attributes: {} (test set)'.format(X.shape[0], X.shape[1]))			

				results = predict_separate(X, y, fname, out_dir+'/'+alg+'_test/', record_id, target_id, feature_selection, model, best_features)
				results_list2.append(results)

		try:
			in_out.save_ROC(out_dir+'/'+alg+'/'+"roc.png", results_list, title='ROC curve')
		except IndexError:
			pass
		
		try:
			in_out.save_ROC(out_dir+'/'+alg+'_test/'+"roc.png", results_list2, title='ROC curve')
		except NameError:
			pass

	# notify user
	print ('## Learning Finished ##')

def execute_with_algorithm(alg, X, y, fname, headers, out_dir, record_id, target_id, feature_selection, oversampling, survival, undersampling, aggregation):
	'''execute learning task using the specified algorithm'''

	# feature selection
	if survival == True and aggregation == True:
		k=150
	if survival == True and aggregation == False:
		k=220
	if survival == False and aggregation == True:
		k=150
	if survival == False and aggregation == False:
		k=220

	new_X, best_features, headers = pearson_fs(X, y, k, headers, feature_selection, survival)

	if aggregation == True:
		new_X = new_X[:,0:-1]
		headers = headers[0:-1]

	# execute algorithm
	if alg == 'DT':
		results, model = ML.CART(new_X, y, best_features, out_dir+"{}.dot".format(fname), headers, oversampling, undersampling, aggregation)  #out_dir+"{}.dot".format(fname)
	elif alg == 'RF':
		results, features, model = ML.RF(new_X, y, best_features,oversampling, undersampling, aggregation, n_estimators=200)
	elif alg == 'RFsmall':
		results, features, model = ML.RF(new_X, y, best_features, oversampling, undersampling, aggregation, n_estimators=100)
	elif alg == 'SVM':
		results, model = ML.SVM(new_X, y, best_features, oversampling, undersampling, aggregation)
	elif alg == 'LR':
		results, features, model = ML.LR(new_X, y, best_features,oversampling, undersampling, aggregation)
	elif alg == 'XGBoost':
		results, features, model = ML.XGBoost(new_X, y, best_features,oversampling, undersampling, aggregation)
	if alg == 'COX':
		results, features, model = ML.COX(new_X, y, best_features, oversampling, undersampling, aggregation)
	if alg == 'survSVM':
		results, features, model = ML.survSVM(new_X, y, best_features, oversampling, undersampling, aggregation)
	if alg == 'GBS':
		results, features, model = ML.GradientBoostingSurvival(new_X, y, best_features, oversampling, undersampling, aggregation)

	if not results:
		return

	# set2model_instance[fname] = (model, best_features)

	# export results
	# results_list.append([fname] + results[0:3])

	if survival == False:
		in_out.save_results(out_dir+fname+'.csv', ["fpr", "tpr", "auc", "cm"], results, [sum(y),len(y)])
	# else:
		# in_out.save_results(out_dir+fname+'.csv', ["CI"], results, [sum(y),len(y)])

	if 'features' in locals():
		features = features.flatten()
		in_out.save_features(out_dir+"features_" + fname + '.csv', zip(headers[1:-1], features))
	
	return model, best_features, [fname] + results[0:3]

def predict_separate(X, y, fname, out_dir, record_id, target_id, feature_selection, model, best_features):
	'''execute learning task using the specified algorithm'''
	print ('  ...testing on new data')
	
	# select the feature selected attribute only
	if best_features == 'all':
		new_X = X
	else:
		new_X = X[:,best_features]

	# execute algorithm
	y_pred = model.predict_proba(new_X)
	fpr, tpr, _ = roc_curve(y, y_pred[:, 1])
	mean_fpr = np.linspace(0, 1, 100)
	mean_tpr = interp(mean_fpr, fpr, tpr)
	mean_auc = auc(fpr, tpr)
	results = [mean_fpr, mean_tpr, mean_auc, np.zeros((2,2))]
	in_out.save_results(out_dir+fname+'.csv', ["fpr", "tpr", "auc", "cm"], results, [sum(y),len(y)])

	results = [fname] + results[0:3]
	return results


def pearson_fs(X, y, k, headers, feature_selection, survival):

	k = k
	if feature_selection: #and X.shape[1] >= k
		print ('  ...performing pearson feature selection')
		
		if survival:
			pearson_y = []
			for tup in y:
				if tup[0] == False:
					pearson_y.append(0)
				else:
					pearson_y.append(1)
		else:
			pearson_y = y

		pearsons = []
		for i in range(X.shape[1]):
			p = pearsonr(np.squeeze(np.asarray(X[:,i])), pearson_y)
			pearsons.append(abs(p[0]))
		best_features = np.array(pearsons).argsort()[-k:][::-1]
		# print best_features

		new_headers = []
		test_list = best_features.tolist()
		for i in best_features:
			if headers[i].upper()[0:3] in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']:
				print(headers[i])
				index = test_list.index(i)
				best_features = np.delete(best_features, index)
				test_list.pop(index)
				continue
			else:
				new_headers.append(headers[i])


		headers = new_headers
		# headers = [headers[i] for i in best_features]
		print('ik blijf maar printen')
		new_X = X[:,best_features]
		# print new_X.shape
		# print y.shape

	else:
		new_X = X
		best_features='all'
		
	return new_X, best_features, headers

def Kbest_fs(X, y, headers, feature_selection, survival):
	k = 70

	if feature_selection and X.shape[1] >= k:
		print ('  ...performing RFE feature selection')
		test = SelectKBest(score_func=chi2, k=k)
		fit = test.fit(X, y)
		new_X = fit.transform(X)

		index = 0
		features = dict()
		scores = fit.scores_

		for header in headers:
			features[header] = scores[index]
			index +=1

		sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
		top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89']][0:k]

		f = open('/Users/Tristan/Downloads/merged/important_features/kbest.txt', 'w')
		simplejson.dump(top_features, f)
		f.close()

		new_headers = [header for header in headers if header in top_features]

	else:
		new_X = X
		new_headers = headers
		top_features = 'all'

	return new_X, top_features, new_headers

def random_forest_fs(X, y, headers, feature_selection):

	k=50
	if feature_selection and X.shape[1] >= k:
		print ('  ...performing random forest regressor feature selection')

		rf = RandomForestRegressor()
		rf.fit(X, y)

		print ("Features sorted by their score:")
		print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), headers), reverse=True))

		best_features = rf.feature_importances_.argsort()[::-1][:k]
		headers = [headers[i] for i in best_features]
		new_X = X[:,best_features]

	else:
		new_X = X
		best_features ='all'


def aggregations(f, target_list, survival):
	rows = read_csv(f)
	headers = rows.columns.values.tolist()

	index_dict_old = dict()
	index_dict_new = dict()
	new_headers = []
	index = 0

	for header in headers:
		key = header.split('_')
				
		if key[0] in ['i', 'd', 's', 'h', 'l', 'n', 'N','L','H','S','D','I']:
			print(key[1])
			if key[1][0:3] in ['rrd', 'rrs', 'RRD', 'RRS']:
				bp_key = key[1][0:3]
				print(bp_key)
				print('uhuhhhhhhhh')
				if bp_key not in index_dict_old:
					index_dict_old[bp_key] = []
					index_dict_new[bp_key] = []

				index_dict_old[bp_key].append(headers.index(header))
				index_dict_new[bp_key].append(headers.index(header))
				index +=1

				if bp_key not in new_headers:
					new_headers.append(key[1])
			
			else:
				if key[1] not in index_dict_old:
					index_dict_old[key[1]] = []
					index_dict_new[key[1]] = []

				index_dict_old[key[1]].append(headers.index(header))
				index_dict_new[key[1]].append(headers.index(header))
				index +=1

				if key not in new_headers:
					new_headers.append(key[1])

		else:
			index_dict_new[header] = []
			index_dict_new[header].append(headers.index(header))
			new_headers.append(header)
			index+=1


	new_rows =  []
	rows = read_csv(f)
	headers = rows.columns.values.tolist()

	for row in tqdm(rows.itertuples()):
		row = row[1:]
		new_row = dict()
		for index, values in index_dict_new.items():
			for value in values:
				try:
					if int(row[value]) >=1:
						new_row[index] = int(row[value])
					else:
						new_row[index] = 0
				except TypeError:
					row_index = row[value]
				except ValueError:
					row_index = row[value]

		value_list = []
		columns_list = []

		for key, value in new_row.items():
			value_list.append(value)
			columns_list.append(key)
        
		new_headers = columns_list
		new_rows.append(value_list)

	records = [row for row in new_rows]
	X = np.matrix(records, dtype='float64')

	if survival == True:
    		X = np.delete(X, (target_list), axis=0)

	headers = new_headers

    	# output

	print(X, headers)
	return X, headers


def read_csv(f, delim=',', index_col='ID'):
	'''opens a csv reader object'''
	# return csv.reader(open(f, 'r'), delimiter=delim)
	return pd.read_csv(f, sep=',', index_col=index_col, encoding = "ISO-8859-1") #index_col='pseudopatnummer'
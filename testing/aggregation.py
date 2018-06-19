import csv

from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import simplejson
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *

def read_csv2(f, delim=','): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def read_csv(f, delim=',', index_col='ID'):
	'''opens a csv reader object'''
	# return csv.reader(open(f, 'r'), delimiter=delim)
	return pd.read_csv(f, sep=',', index_col=index_col, encoding = "ISO-8859-1") #index_col='pseudopatnummer'

def get_headers(row):
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
    print(X)
    headers = headers[1:-1]

	# output
    y = records[:,-1] # target

    if survival == False:
        y=np.squeeze(np.asarray(y.astype(np.int)))

        print ('  ...(converting data type)')

        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)

        print(y)
        print(X)
	
    if survival == True:
        target_list = []

        y=np.squeeze(np.asarray(y.astype(list)))
        X = X.astype(np.float64, copy=False)

        for target in y:
            target = eval(target)
            tuple_target = tuple(target)
            target_list.append(tuple_target)

		# print(target_list)
        y = np.array(target_list, dtype=[('Status', '?'), ('Survival in days', '<f8')])

        print('dope')
        print(y)
        print(X)
	
        print ('  ...(converting data type)')
		# X = X.astype(np.float64, copy=False)
		# print(X)
		# X = OneHotEncoder().fit_transform(X)


    return X, y, headers


f = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/survival/survival.csv'#sys.argv[1]
target_id = 'target'
record_id = 'x'
survival = True

# X,y, headers = import_data(f, record_id, target_id, survival=True)

def aggregation(f):
    rows = read_csv(f)
    headers = rows.columns.values.tolist()

    index_dict_old = dict()
    index_dict_new = dict()
    new_headers = []
    index = 0

    for header in headers:
        key = header.split('_')

        if key[0] in ['i', 'd', 's', 'h', 'l', 'n', 'N','L','H','S','D','I']:

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


    new_df = pd.DataFrame()#columns=new_headers
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

    checklist=[]
    for row in new_rows:
        if str(len(row)) not in checklist:
            checklist.append(str(len(row)))
            print(str(len(row)))

    records = [row for row in new_rows]
    X = np.asarray(records, dtype='float64')
    print(X)
    headers = new_headers

    # output

    print ('  ...(converting data type)')

    # X = X.astype(np.float64, copy=False)

    print(X, headers)
    # return X, headers





    
# print(headers.index(header))
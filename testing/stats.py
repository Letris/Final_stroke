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

def read_csv2(f, delim=','): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def get_headers(row):
	'''returns the non-capitalised and bugfixed version of the header'''
	headers = [el.lower() for el in row]
	headers[0] = headers[0].split("\xef\xbb\xbf")[1] if headers[0].startswith('\xef') else headers[0] # fix funny encoding problem
	return headers



	return rows, headers

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


f = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/FINAL/survivalFINAL/survivalFINAL.csv'#sys.argv[1]
target_id = 'target'
record_id = 'x'
survival = True

X,y, headers = import_data(f, record_id, target_id, survival=True)


dic = dict()
dic[1] = []
dic[0] = []
stroke = 0
no_stroke = 0

for target in y:
    if target[0] == True:
        stroke+=1
        dic[1].append(target[1])
    else:
        no_stroke+=1
        dic[0].append(target[1])
    
print(stroke,no_stroke)
mean_pos = np.mean(dic[1])
mean_neg = np.mean(dic[0])

avg_ns = np.mean(dic[0])
avg_s = np.mean(dic[1])

print(avg_ns, avg_s)

plt.hist(dic[1]) 
# plt.axis([0, 5000, 0, 50000])
plt.title('Survival time frequencies stroke')
plt.ylabel('Frequency')
plt.xlabel('Survival time')
plt.show()


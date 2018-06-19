# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, make_scorer
from scipy.stats import pearsonr
import matplotlib.pylab as plt
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import csv
import sys
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from imblearn.over_sampling import SMOTE
from scipy import interp
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn import ensemble, svm, tree, linear_model
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
import simplejson
from skopt import BayesSearchCV
from sklearn.cross_validation import StratifiedKFold
from tqdm import *
import pandas as pd

def read_csv(f, delim=','): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def read_csvpd(f, delim=',', index_col='ID'):
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

    rows = read_csv(f)
    headers = get_headers(next(rows))

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
                print('yes sir')
                index_list.append(idx)
                continue

            target_list.append(tuple_target)

        
		# print(target_list)
        y = np.array(target_list, dtype=[('Status', '?'), ('Survival in days', '<f8')])

        X = np.delete(X, (index_list), axis=0)
        # y = np.delete(y, (index_list), axis=0)

        print ('  ...(converting data type)')
		# X = X.astype(np.float64, copy=False)
		# print(X)
		# X = OneHotEncoder().fit_transform(X)


    return X, y, headers, index_list


def Kbest_fs(X, y, headers, k, feature_selection, survival):
    if feature_selection and X.shape[1] >= k:
        print ('  ...performing RFE feature selection')

    if survival:
        kbest_y = []
        for tup in y:
            if tup[0] == False:
                kbest_y.append(0)
            else:
                kbest_y.append(1)
    else:
        kbest_y = y

    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(X, kbest_y)
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

    return new_X, top_features



def pearson_fs(X, y, headers, k, feature_selection, survival):

    if feature_selection:
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
        print(best_features)
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

        print(headers)
        new_X = X[:,best_features]
        f = open('/Users/Tristan/Downloads/merged/important_features/pearson.txt', 'w')
        simplejson.dump(headers, f)
        f.close()

    else:
        new_X = X
        best_features = 'all'
    
    return new_X, best_features

def CI(y_true, y_pred):
    event_indicators = []
    event_times = []
    scores = []

    for target in y_true:
        event_indicators.append(target[0])
        event_times.append(target[1])

        # print(X[test].shape, y[test].shape)
        # CIscore +=trained_classifier.score(X[test], y[test])

    for prediction in y_pred:
        scores.append(prediction)
        # print(prediction)

    result = concordance_index_censored(np.array(event_indicators), np.array(event_times), np.array(scores).reshape(-1))[0]

    return result

def aggregations(f, target_list, survival):
    rows = read_csvpd(f)
    headers = rows.columns.values.tolist()

    index_dict_old = dict()
    index_dict_new = dict()
    new_headers = []
    index = 0

    for header in headers:
        key = header.split('_')
        bp_key = header[0:2]

        if key[0] in ['i', 'd', 's', 'h', 'l', 'n', 'N','L','H','S','D','I']:

            if key[1] not in index_dict_old:
                index_dict_old[key[1]] = []
                index_dict_new[key[1]] = []

            index_dict_old[key[1]].append(headers.index(header))
            index_dict_new[key[1]].append(headers.index(header))
            index +=1

            # if key not in new_headers:
            #     new_headers.append(key[1])

        elif bp_key in ['rrd', 'rrs', 'RRD', 'RRS']:

            if bp_key not in index_dict_old:
                index_dict_old[bp_key] = []
                index_dict_new[bp_key] = []

            index_dict_old[bp_key].append(headers.index(header))
            index_dict_new[bp_key].append(headers.index(header))
            index +=1

          
        else:
            index_dict_new[header] = []
            index_dict_new[header].append(headers.index(header))
            # new_headers.append(header)
            index+=1


    new_rows =  []
    rows = read_csvpd(f)
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
    X = np.matrix(records, dtype='float64')

    if survival == True:
        X = np.delete(X, (target_list), axis=0)

    headers = new_headers

    # output

    print ('  ...(converting data type)')

    # X = X.astype(np.float64, copy=False)

    return X, headers

def RandomGridSearchRFC_Fixed(X,Y,splits, model, survival):
    """
    This function looks for the best set o parameters for RFC method
    Input: 
        X: training set
        Y: labels of training set
        splits: cross validation splits, used to make sure the parameters are stable
    Output:
        clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
    """    
      

    start_svm = time.time()  
    
    if model == 'svm':
        clf = svm.SVC()

        tuned_parameters = {
        'C': ([0.01, 1, 10]),
         'kernel': (['rbf', 'linear']),
        # 'kernel': (['linear', 'rbf', 'sigmoid']),
        # 'degree': ([1,3,5,10]),
        # 'decision_function_shape' : (['ovo', 'ovr']),
        # 'cache_size': ([500,1000,1500,2000]),
        'shrinking': ([False, True]),
        # 'probability': ([False, True])
        }
    
    if model == 'cart':
        clf = tree.DecisionTreeClassifier()

        tuned_parameters = {
        'criterion': (['gini', 'entropy']),
        'max_depth': ([10,20]),
        'min_samples_split': ([2,3,5]),
        'min_samples_leaf': ([2,3,5]),
        }

    if model == 'rf':
        clf = ensemble.RandomForestClassifier()
 
        tuned_parameters = {
        'n_estimators': ([200,500,1000]),
        # 'max_features': (['auto', 'sqrt', 'log2',1,4,8]),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20]),
        # 'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,3,5],
        'min_samples_leaf':   [2,3,5],
        }
        
    if model == 'xgboost':
        clf = XGBClassifier()

        tuned_parameters = {
        'booster': (['gbtree']),
        'max_depth':   ([5,10,20]),
        'reg_lambda': ([0,1]),
        'reg_alpha': ([0,1]),
        'subsample': ([0.5,1])
        }

    if model == 'lr':
        clf = linear_model.LogisticRegression()

        tuned_parameters = {
        'solver': (['liblinear', 'sag', 'saga'])
        }

    if model == 'cox':
       
        clf =  CoxnetSurvivalAnalysis()
        tuned_parameters = {
        'n_alphas': ([50,100,200]),
        'l1_ratio': ([0.1,0.5,1]),

        }

    if model == 'survSVM':
        clf = FastSurvivalSVM()
        
        tuned_parameters = {
        'alpha': ([0.5,1]),
        'rank_ratio': ([0.5,1]),
        'max_iter': ([20,40,80]),
        'optimizer': (['rbtree', 'avltree']),
        }

    if model == 'gb':
        clf = GradientBoostingSurvivalAnalysis()
       
        tuned_parameters = {
        'learning_rate': ([0.1, 0.3]),
        'n_estimators': ([100,200,400]),
        'max_depth': ([3,6,12])        
        }

    
    if survival == True:
        scorer = make_scorer(CI, greater_is_better=True)

        y_for_cv = np.array([t[0] for t in Y])
        cv = StratifiedKFold(y_for_cv, n_folds=2) # x-validation

    else:
        cv = StratifiedKFold(Y, n_folds=2) # x-validation
        scores = ['roc_auc']   

    print ('  ...performing x-validation')
   
    clf =  GridSearchCV(clf, tuned_parameters, scoring='%s' % scores[0], cv=cv, verbose=10) #scoring='%s' % scores[0]
    # clf = BayesSearchCV(clf, tuned_parameters, n_iter=50, cv=splits,
    #                 optimizer_kwargs=dict(acq_func='LCB', base_estimator='RF'))

    clf.fit(X, Y)

    end_svm = time.time()
    print("Total time to process: ",end_svm - start_svm)
  
    return(clf.best_params_,clf)

sm = SMOTE(random_state=12, ratio = 1.0)

f = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/nonsurvFINAL/nonsurvFINAL.csv'#sys.argv[1]
target_id = 'target'
record_id = 'x'
survival = False
aggregation = True
k=150

x, y, headers, index_list = import_data(f, record_id, target_id, survival) # assumption: first column is patientnumber and is pruned, last is target

if aggregation == True:
    X, headers = aggregations(f, index_list, survival)

new_X, best_features = pearson_fs(X, y, headers, k, feature_selection=True, survival=False)

m_dict = dict()

for m in ['cart', 'svm', 'rf', 'xgboost', 'lr']: #'cart', 'svm', 'rf', 'xgboost', 'lr'
    best_params, model = RandomGridSearchRFC_Fixed(new_X, y, 2, m, survival)
    m_dict[m] = best_params
    print(m)
    print(best_params)
    print(m_dict)

print(m_dict)



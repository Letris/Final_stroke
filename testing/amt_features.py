# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import pearsonr
from scipy import interp
import matplotlib.pylab as plt
import pandas as pd
import csv
from sklearn import ensemble, svm, tree, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import simplejson
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from tqdm import *
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sklearn import ensemble, svm, tree, linear_model

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

def write_csv(f):
	'''opens a csv writer object'''	
	return csv.writer(open(f,"w"))

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


def pearson_fs(X, y, headers, k, feature_selection, survival):

    if feature_selection:
        # print ('  ...performing pearson feature selection')
		
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
        print(pearsons)
        best_features = np.array(pearsons).argsort()[-k:][::-1]
        print(best_features)
		# print best_features
        print(headers[best_features])
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


def lasso_fs(X, y, headers, k, feature_selection):
    print ('  ...performing lasso feature selection')
    if feature_selection and X.shape[1] >= k:
       
        lasso = Lasso()
        lasso.fit(X, y)
        scores = lasso.coef_

        index = 0
        features = dict()
        to_indeces = []

        for header in headers:
            features[header] = scores[index]
            index +=1

        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']][0:k]
        print(top_features)

        for feature in top_features:
            if feature in headers:
                to_indeces.append(headers.index(feature))
        
        headers = top_features
        new_X = X[:,to_indeces]

        f = open('/Users/Tristan/Downloads/merged/important_features/lasso.txt', 'w')
        simplejson.dump(top_features, f)
        f.close()

        return new_X, top_features


def random_forest_fs(X, y, headers, k, feature_selection):

    if feature_selection and X.shape[1] >= k:
        print ('  ...performing random forest regressor feature selection')

        rf = RandomForestRegressor()
        rf.fit(X, y)

        print ("Features sorted by their score:")
        print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), headers), reverse=True))

        best_features = rf.feature_importances_.argsort()[::-1][:k]

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

        # headers = [headers[i] for i in best_features]

        headers = new_headers
        new_X = X[:,best_features]

    else:
        new_X = X
        best_features ='all'

    f = open('/Users/Tristan/Downloads/merged/important_features/RF.txt', 'w')
    simplejson.dump(headers, f)
    f.close()
    print(headers)
    return new_X, best_features


def RFE_fs(X, y, headers, k, feature_selection):

    if feature_selection and X.shape[1] >= k:
        print ('  ...performing RFE feature selection')

        model = linear_model.LogisticRegression()
        rfe = RFE(model, k)
        rfe.fit(X, y)

        index = 0
        new_headers = []
        for header in headers:
            if rfe.support_[index] == True and header.upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']:
                new_headers.append(header)
                index +=1
            else:
                index +=1

        headers = new_headers
        new_X = X[headers]
       
        f = open('/Users/Tristan/Downloads/merged/important_features/RFE.txt', 'w')
        simplejson.dump(headers, f)
        f.close()

    print(headers)
    return new_X, headers


def Kbest_fs(X, y, headers, k, feature_selection):
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
    top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'tar', 'TAR', 'target', 'TARGET']][0:k]

    f = open('/Users/Tristan/Downloads/merged/important_features/kbest.txt', 'w')
    simplejson.dump(top_features, f)
    f.close()

    return new_X, top_features


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

	
        print ('  ...(converting data type)')
		# X = X.astype(np.float64, copy=False)
		# print(X)
		# X = OneHotEncoder().fit_transform(X)


    return X, y, headers, index_list

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


def execute_nonsurvival(X, y, k, headers, survival, aggregation):
        new_X, best_features = pearson_fs(X, y, headers, k, feature_selection=True, survival=survival)
        n_estimators=200
        cv = StratifiedKFold(y, n_folds=3) # x-validation 

        if aggregation == True:
            lr = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=20, min_samples_leaf=5, min_samples_split=5, n_jobs=-1)
        else:
            lr = ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, min_samples_split=2, n_jobs=-1)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        cm=np.zeros((2,2))

        # cross fold validation
        print ('  ...performing x-validation')
        for i, (train, test) in enumerate(cv):
                print ('   ...',i+1)
                if sum(y[train]) < 5:
                    print ('...cannot train; too few positive examples')

                
                # oversampling with SMOTE
            
                sm = SMOTE(random_state=12, ratio = 1.0)
                x_train, y_train = sm.fit_sample(new_X[train], y[train])


                trained_classifier = lr.fit(x_train, y_train)
                
                y_pred = trained_classifier.predict_proba(new_X[test])
            
                # make cutoff for confusion matrix
                y_pred_binary = (y_pred[:,1] > 0.01).astype(int)

                # derive ROC/AUC/confusion matrix
                fpr, tpr, thresholds = roc_curve(y[test], y_pred[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                cm = cm + confusion_matrix(y[test], y_pred_binary) 

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print(mean_auc)
        mean_cm = cm/len(cv)

        return mean_auc

def execute_survival(X, y, k, headers, survival, aggregation):
    new_X, best_features = pearson_fs(X, y, headers, k, feature_selection=True, survival=survival)
    y_for_cv = np.array([t[0] for t in y])
    cv = StratifiedKFold(y_for_cv, n_folds=5) # x-validation

    if aggregation == True:
        clf = CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=100)
    else: 
        clf = CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200)

    CIscore = 0
    
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)
    
        y_train = y[train]
        trained_classifier = clf.fit(new_X[train], y[train])


        event_indicators = []
        event_times = []
        scores = []

        for target in y[test]:
           event_indicators.append(target[0])
           event_times.append(target[1])

        predictions = trained_classifier.predict(new_X[test])

        for prediction in predictions:
            scores.append(prediction)
        # print(prediction)

        result = concordance_index_censored(np.array(event_indicators), np.array(event_times), np.array(scores).reshape(-1))
        CIscore += result[0]
        # TODO fix metrics

    avgCIscore = CIscore / len(cv)
    print(avgCIscore)

    return avgCIscore


def amt_features(f, aggregation, survival):
    record_id = 'ID'
    target_id = 'target'

    X, y, headers, target_list = import_data(f, record_id, target_id, survival) # assumption: first column is patientnumber and is pruned, last is target
    print ('  ...instances: {}, attributes: {}'.format(X.shape[0], X.shape[1]))

    if aggregation == True:
        print('woosj')
        X, headers = aggregations(f, target_list, survival)

    # lr = linear_model.LogisticRegression()

    auc_k_dict_pearson = dict()
    best_k = 0
    best_AUC = 0
    best_IC = 0

    for k in range(25,1000,25):
        k=200
        # y_for_cv = np.array([t[0] for t in y])
        if survival == True:
            mean_IC = execute_survival(X, y, k, headers, survival, aggregation)
            auc_k_dict_pearson[k] = mean_IC
            if mean_IC > best_IC:
                best_k = k
        else:
            mean_AUC = execute_nonsurvival(X, y, k, headers, survival, aggregation)
            auc_k_dict_pearson[k] = mean_AUC
            if mean_AUC > best_AUC:
                best_k = k
            
           
    lists_pearson = sorted(auc_k_dict_pearson.items()) # sorted by key, return a list of tuples
    
    x_pearson, y_pearson = zip(*lists_pearson) # unpack a list of pairs into two tuples
    
    print('best k is: {}'.format(best_k))

    return x_pearson, y_pearson

f_surv = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/survivalFINAL/survivalFINAL.csv'
f_nonsurv = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/FINAL/nonsurvFINAL/nonsurvFINAL.csv'

x_nonsurv, y_nonsurv = amt_features(f_nonsurv, aggregation=False, survival=False)
x_surv_agg, y_surv_agg = amt_features(f_surv, aggregation=True, survival=True)
x_surv, y_surv = amt_features(f_surv, aggregation=False, survival=True)
x_nonsurv_agg, y_nonsurv_agg = amt_features(f_nonsurv, aggregation=True, survival=False)


plt.plot(x_surv, y_surv, label='Survival')
plt.plot(x_surv_agg, y_surv_agg, label='Survival + Aggregation')
plt.plot(x_nonsurv, y_nonsurv, label='Non-survival')
plt.plot(x_nonsurv_agg, y_nonsurv_agg, label='Non-survival + Aggregation')
plt.title('K-AUC trade-off curves ')
plt.ylabel('AUC')
plt.xlabel('Number of features')
plt.legend()
plt.savefig('/Users/Tristan/Downloads/TEST_AUC_plot.png')
plt.show()


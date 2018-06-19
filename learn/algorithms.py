# os-related imports
import sys
import csv
import time

# numpy
import numpy as np

# algorithms
from sklearn import ensemble, svm, tree, linear_model

# statistics, metrics, x-fold val, plots
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from imblearn.over_sampling import SMOTE
from scipy import interp
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import scipy as sp
import scipy.stats

# non-survival models

def SVM(X, y, best_features, oversampling, undersampling, aggregation):
    if aggregation == True:
        clf = svm.SVC(probability=True, cache_size=1500, verbose=True, shrinking=False, C=0.01, kernel='rbf') #shrinking=False, probability=True, cache_size=1500,decision_function_shape='ovo', degree=1, kernel='linear'
    else:
        clf = svm.SVC(probability=True, cache_size=1500, verbose=True, shrinking=False, C=0.001, kernel='linear') #probability=True, cache_size=1500, verbose=True, shrinking=False, C=0.001, kernel='linear'

    # e_clf = ensemble.BaggingClassifier(clf, n_estimators=1, max_samples = 0.2, n_jobs=-1, verbose=True)
    results, model = execute(X, y, best_features, lambda: clf, oversampling, undersampling)
    return results, model

def CART(X, y, best_features, out_file, field_names, oversampling, undersampling, aggregation):
    if aggregation == True:
        results, model = execute(X, y, best_features, lambda: tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=5, min_samples_split=5), oversampling, undersampling)
    else:
        results, model = execute(X, y, best_features, lambda: tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=2, min_samples_split=2), oversampling, undersampling)
    if model:
        tree.export_graphviz(model, out_file=out_file, feature_names=field_names)
    return results, model

def RF(X, y, best_features, oversampling, undersampling, aggregation, n_estimators):
    if aggregation == True:
        results, model = execute(X, y, best_features, lambda: ensemble.RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, min_samples_split=2, n_jobs=-1), oversampling, undersampling)
    else:
        results, model = execute(X, y, best_features, lambda: ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=3, min_samples_split=5, n_jobs=-1), oversampling, undersampling)
    if model:
        features = model.feature_importances_
    else:
        features = False
    return results, features, model
   
def LR(X, y, best_features, oversampling, undersampling, aggregation):
    if aggregation == True:
        results, model = execute(X, y, best_features, lambda: linear_model.LogisticRegression(solver='liblinear', max_iter=10000), oversampling, undersampling)
    else:
        results, model = execute(X, y, best_features, lambda: linear_model.LogisticRegression(solver='liblinear', max_iter=10000), oversampling, undersampling)
    if model:
        features = model.coef_
    else:
        features = False
    return results, features, model

def XGBoost(X, y, best_features, oversampling, undersampling, aggregation):
    if aggregation == True:
        results, model = execute(X, y, best_features, lambda: XGBClassifier(booster='gbtree', max_depth=20, reg_alpha=1, reg_lambda=1, subsample=0.5), oversampling, undersampling)
    else:
        results, model = execute(X, y, best_features, lambda: XGBClassifier(booster='gbtree', max_depth=10, reg_alpha=1, reg_lambda=1, subsample=1), oversampling, undersampling)
    if model:
        features = model.feature_importances_
    else:
        features = False
    return results, features, model

# survival models
def COX(X, y, best_features, oversampling, undersampling, aggregation):
    if aggregation == True:
        results, model = execute_survival(X, y, best_features, lambda: CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=100), oversampling, undersampling)
    else:
        results, model = execute_survival(X, y, best_features, lambda: CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200), oversampling, undersampling)
    if model:
        features = model.coef_
    else:
        features = False
    return results, features, model

def survSVM(X, y, best_features, oversampling, undersampling, aggregation):
    if aggregation == True:
        results, model = execute_survival(X, y, best_features, lambda:FastSurvivalSVM(optimizer="rbtree", max_iter=1000, tol=1e-6, random_state=0, alpha=1, rank_ratio=1), oversampling, undersampling)
    else:
        results, model = execute_survival(X, y, best_features, lambda:FastSurvivalSVM(optimizer="rbtree", max_iter=1000, tol=1e-6, random_state=0, rank_ratio=1), oversampling, undersampling)
    if model:
        features = model.coef_
    else:
        features = False
    return results, features, model

def GradientBoostingSurvival(X, y, best_features, oversampling, undersampling, aggregation):
    results, model = execute_survival(X, y, best_features, lambda:GradientBoostingSurvivalAnalysis(learning_rate=0.1, max_depth=3, n_estimators=100), oversampling, undersampling)
    if model:
        features = model.feature_importances_
    else:
        features = False
    return results, features, model


def hyperparameter_tuning(model, params, X, y):
    # tune the hyperparameters via a randomized search
    grid = RandomizedSearchCV(model, params)
    start = time.time()
    grid.fit(X, y)
    
    # evaluate the best randomized searched model on the testing
    # data
    print("[INFO] randomized search took {:.2f} seconds".format(
        time.time() - start))
    acc = grid.score(X, y)
    print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
    print("[INFO] randomized search best parameters: {}".format(
        grid.best_params_))

def execute_survival(X, y, best_features, classifier, oversampling, undersampling):
    y_for_cv = np.array([t[0] for t in y])
    cv = StratifiedKFold(y_for_cv, n_folds=5) # x-validation
    classifier = classifier()

    clf = Pipeline([('classifier',classifier)])

    CIscore = 0
    score_list = []
    
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)

        total = 0
        for target in y[train]:
            if target[0] == True:
                total += 1

        if total < 5:
            print ('...cannot train; too few positive examples')
            return False, False

        if oversampling == True:
            sm = SMOTE(random_state=12, ratio = 1.0)
            x_train, y_train = sm.fit_sample(X[train], y[train])
            trained_classifier = clf.fit(x_train, y_train)
        if undersampling ==  True:
            rus = RandomUnderSampler(random_state=0)
            x_train, y_train = rus.fit_sample(X[train], y[train])
            trained_classifier = clf.fit(x_train, y_train)
        else:
            y_train = y[train]
            trained_classifier = clf.fit(X[train], y[train])


        event_indicators = []
        event_times = []
        scores = []

        for target in y[test]:
           event_indicators.append(target[0])
           event_times.append(target[1])

        # print(X[test].shape, y[test].shape)
        # CIscore +=trained_classifier.score(X[test], y[test])

        predictions = trained_classifier.predict(X[test])

        for prediction in predictions:
            scores.append(prediction)
        # print(prediction)
        print(np.array(event_indicators).shape, np.array(event_times).shape, np.array(scores).reshape(-1).shape)

        result = concordance_index_censored(np.array(event_indicators), np.array(event_times), np.array(scores).reshape(-1))
        print(result[0])
        CIscore += result[0]
        score_list.append(result[0])
        # TODO fix metrics

    avgCIscore = CIscore / len(cv)
    print('done yas')
    print(avgCIscore)

    low_bound, high_bound = confidence_interval(score_list)

    print ('  ...fitting model (full data sweep)'.format(X.shape))

    complete_classifier = clf.fit(X,y)

    print('it worked')

    return avgCIscore, complete_classifier.named_steps['classifier']


def execute(X, y, best_features, classifier, oversampling, undersampling):
    cv = StratifiedKFold(y, n_folds=5) # x-validation
    classifier = classifier()
    
        
    # print np.var(X[:,0]),np.var(X[:,1]),np.var(X[:,2]),np.var(X[:,3]),np.var(X[:,4]),np.var(X[:,5]),

    clf = Pipeline([('classifier',classifier)])

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    cm=np.zeros((2,2))
    score_list = []
    # cross fold validation
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)
        if sum(y[train]) < 5:
            print ('...cannot train; too few positive examples')
            return False, False

        # train
        # if type(clf.named_steps['classifier']) == tree.DecisionTreeClassifier:
        #     num_pos = sum(y[train]) # number of positive cases
        #     w0 = num_pos / len(y[train]) # inversely proportional with number of positive cases
        #     w1 = 1 - w0 # complement of w0
        #     sample_weight = np.array([w0 if el==0 else w1 for el in y[train]])
        #     trained_classifier = clf.named_steps['classifier'].fit(X[train], y[train], sample_weight=sample_weight)
        #     #trained_classifier = clf.fit(X[train], y[train], sample_weight=sample_weight)
        # else:
        #     trained_classifier = clf.fit(X[train], y[train])

        # oversampling with SMOTE
        if oversampling == True:
            sm = SMOTE(random_state=12, ratio = 1.0)
            x_train, y_train = sm.fit_sample(X[train], y[train])
            trained_classifier = clf.fit(x_train, y_train)
        if undersampling ==  True:
            rus = RandomUnderSampler(random_state=0)
            x_train, y_train = rus.fit_sample(X[train], y[train])
            trained_classifier = clf.fit(x_train, y_train)
        else:
           trained_classifier = clf.fit(X[train], y[train])


        y_pred = trained_classifier.predict_proba(X[test])
       

        # make cutoff for confusion matrix
        y_pred_binary = (y_pred[:,1] > 0.01).astype(int)

        # derive ROC/AUC/confusion matrix
        fpr, tpr, thresholds = roc_curve(y[test], y_pred[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        cm = cm + confusion_matrix(y[test], y_pred_binary) 
        score_list.append(auc(fpr, tpr))

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_cm = cm/len(cv)

    low_bound, high_bound = confidence_interval(score_list)

    # redo with all data to return the features of the final model
    print ('  ...fitting model (full data sweep)'.format(X.shape))
    complete_classifier = clf.fit(X,y)
    print(mean_fpr, mean_tpr, mean_auc, mean_cm)
    return [mean_fpr, mean_tpr, mean_auc, mean_cm], complete_classifier.named_steps['classifier']

def read_csv(f, delim=','): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def get_headers(row):
	'''returns the non-capitalised and bugfixed version of the header'''
	headers = [el.lower() for el in row]
	headers[0] = headers[0].split("\xef\xbb\xbf")[1] if headers[0].startswith('\xef') else headers[0] # fix funny encoding problem
	return headers

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sample_size = len(data)
    std = np.std(data)
    standard_error = std / (np.sqrt(sample_size))
    margin_of_error = float(standard_error * 2)
    low_bound = float(mean - margin_of_error)
    high_bound = float(mean + margin_of_error)

    print(low_bound, high_bound)
    return low_bound, high_bound 

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

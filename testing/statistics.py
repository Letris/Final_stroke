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


f = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/nonsurvFINAL/nonsurvFINAL.csv'#sys.argv[1]
target_id = 'target'
record_id = 'x'
survival = False

# X,y, headers = import_data(f, record_id, target_id, survival=True)

rows = read_csv(f)
headers = rows.columns.values.tolist()

new_headers = []

# for header in headers:
#     try:
#         key = header.split('_')
#         if key[1] in ['ATC', 'ICPC', 'atc', 'icpc']:
#             print(key[1])
#             new_headers.append(key[1])
#         else:
#             new_headers.append(header)
#     except IndexError:
#         new_headers.append(key[0])

# print(new_headers)
# rows.columns = new_headers
mean_age_stroke = []
mean_age_control = []
mean_age = []

amount_men_stroke = 0
amount_women_stroke = 0
amount_men_control = 0
amount_women_control = 0
amount_men = 0
amount_women = 0

set_icpcs_stroke = []
set_icpcs_control = []
set_icpcs = []

amount_icpcs_stroke = 0
amount_icpcs_control = 0
amount_icpcs = 0

set_atcs_stroke = []
set_atcs_control = []
set_atcs = []

amount_atcs_stroke = 0
amount_atcs_control = 0
amount_atcs = 0

total_other = 0
total_lab = 0

amount_lab_stroke = 0
amount_lab_control = 0
amount_lab = 0

amount_other_stroke = 0
amount_other_control = 0
amount_other = 0

set_lab_stroke = []
set_lab_control = []
set_lab = []

set_other_stroke = []
set_other_control = []
set_other = []

other_feats = {}
lab_feats = {}
atc_feats = {}
icpc_feats = {}

total_stroke = 0
total_control = 0

mean_icpcs_per_patient = []
mean_icpcs_per_patient_stroke = []
mean_icpcs_per_patient_control = []

mean_atcs_per_patient = []
mean_atcs_per_patient_stroke = []
mean_atcs_per_patient_control = []

mean_other_per_patient = []
mean_other_per_patient_stroke = []
mean_other_per_patient_control = []

mean_lab_per_patient = []
mean_lab_per_patient_stroke = []
mean_lab_per_patient_control = []

survival = False
t1=0
t2=0
for row in tqdm(rows.itertuples()):
    index = 0
    if survival == True:   
        target=row[-1]  
        target = eval(target)
        tuple_target = tuple(target)
        if tuple_target[0] == False:
            t1+=1
            target = 0
        else:
            t2+=1
            target = 1

    if survival == False:
        target = row[-1]

    if target == 1:
        total_stroke+=1
    else:
        total_control +=1

    total_icpc_per_patient = 0
    total_icpc_per_patient_stroke = 0
    total_icpc_per_patient_control = 0

    total_atc_per_patient = 0
    total_atc_per_patient_stroke = 0
    total_atc_per_patient_control = 0

    total_lab_per_patient = 0
    total_lab_per_patient_stroke = 0
    total_lab_per_patient_control = 0

    total_other_per_patient = 0
    total_other_per_patient_stroke = 0
    total_other_per_patient_control = 0


    # gender = headers.index('gender')

    index= 0
    for attr in row[1:-1]:
        name_attr = headers[index]
        # print(name_attr, attr)
        
        if target == 1:

            if name_attr == 'age':
               mean_age.append(attr)
               mean_age_stroke.append(attr)
               index+=1
               continue

            if name_attr == 'gender' and attr == 0:
                amount_men +=1
                amount_men_stroke +=1
                index+=1
                continue

            if name_attr == 'gender' and attr == 1:
                amount_women +=1
                amount_women_stroke +=1
                index+=1
                continue

            if name_attr.split('_')[1] in ['ICPC', 'icpc']:
                if not name_attr in icpc_feats:
                    set_icpcs.append(name_attr)
                    set_icpcs_stroke.append(name_attr)
                    icpc_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    
                    if attr >=1:
                        icpc_feats[name_attr]['amount_stroke'] += 1
                        icpc_feats[name_attr]['amount'] += 1
                        icpc_feats[name_attr]['possible_amount_stroke'] +=1
                        total_icpc_per_patient +=1
                        total_icpc_per_patient_stroke +=1
                        amount_icpcs_stroke +=1
                        amount_icpcs +=1
                        index+=1
                        continue
                    else:
                        icpc_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            icpc_feats[name_attr]['amount_stroke'] += 1
                            icpc_feats[name_attr]['amount'] += 1
                            icpc_feats[name_attr]['possible_amount_stroke'] +=1
                            total_icpc_per_patient +=1
                            total_icpc_per_patient_stroke +=1
                            amount_icpcs_stroke +=1
                            amount_icpcs +=1
                            index+=1
                            continue
                    else:
                        icpc_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue

            if name_attr.split('_')[1] in ['ATC', 'atc']:
                if not name_attr in atc_feats:
                    set_atcs.append(name_attr)
                    set_atcs_stroke.append(name_attr)
                    atc_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    
                    if attr >=1:
                        atc_feats[name_attr]['amount_stroke'] += 1
                        atc_feats[name_attr]['amount'] += 1
                        atc_feats[name_attr]['possible_amount_stroke'] +=1
                        total_atc_per_patient +=1
                        total_atc_per_patient_stroke +=1
                        amount_atcs_stroke +=1
                        amount_atcs +=1
                        index+=1
                        continue
                    else:
                        atc_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            atc_feats[name_attr]['amount_stroke'] += 1
                            atc_feats[name_attr]['amount'] += 1
                            atc_feats[name_attr]['possible_amount_stroke'] +=1
                            total_atc_per_patient +=1
                            total_atc_per_patient_stroke +=1
                            amount_atcs_stroke +=1
                            amount_atcs +=1
                            index+=1
                            continue
                    else:
                        atc_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue


            if name_attr.split('_')[0] in ['i', 'd', 's', 'h', 'l', 'n', 'N','L','H','S','D','I']:
                if not name_attr in lab_feats:
                    set_lab.append(name_attr)
                    set_lab_stroke.append(name_attr)
                    lab_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    
                    if attr >=1:
                        lab_feats[name_attr]['amount_stroke'] += 1
                        lab_feats[name_attr]['amount'] += 1
                        lab_feats[name_attr]['possible_amount_stroke'] +=1
                        total_lab_per_patient +=1
                        total_lab_per_patient_stroke +=1
                        amount_lab_stroke +=1
                        amount_lab +=1
                        index+=1
                        continue
                    else:
                        lab_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            lab_feats[name_attr]['amount_stroke'] += 1
                            lab_feats[name_attr]['amount'] += 1
                            lab_feats[name_attr]['possible_amount_stroke'] +=1
                            total_lab_per_patient +=1
                            total_lab_per_patient_stroke +=1
                            amount_lab_stroke +=1
                            amount_lab +=1
                            index+=1
                            continue
                    else:
                        lab_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue
                
            else:
                if not name_attr in other_feats:
                    set_other.append(name_attr)
                    set_other_stroke.append(name_attr)
                    other_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}

                    if attr >=1 :
                        other_feats[name_attr]['amount_stroke'] +=1
                        other_feats[name_attr]['amount'] +=1
                        other_feats[name_attr]['possible_amount_stroke'] +=1
                        total_other_per_patient+=1
                        total_other_per_patient_stroke+=1
                        amount_other_stroke +=1
                        amount_other +=1
                        index+=1
                        continue
                    else:
                        other_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1 :
                        other_feats[name_attr]['amount_stroke'] +=1
                        other_feats[name_attr]['amount'] +=1
                        other_feats[name_attr]['possible_amount_stroke'] +=1
                        total_other_per_patient+=1
                        total_other_per_patient_stroke+=1
                        amount_other_stroke +=1
                        amount_other +=1
                        index+=1
                        continue
                    else:
                        other_feats[name_attr]['possible_amount_stroke'] +=1
                        index+=1
                        continue


        if target == 0:

            if name_attr =='age':
                mean_age.append(attr)
                mean_age_control.append(attr)
                index+=1
                continue

            if name_attr == 'gender' and attr == 0:
                amount_men += 1
                amount_men_control +=1
                index+=1
                continue

            if name_attr == 'gender' and attr == 1:
                amount_women +=1
                amount_women_control +=1
                index+=1
                continue

            # if name_attr == ''
                
            if name_attr.split('_')[1] in ['ICPC', 'icpc']:
                if not name_attr in icpc_feats:
                    set_icpcs.append(name_attr)
                    set_icpcs_control.append(name_attr)
                    icpc_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    
                    if attr >=1:
                        icpc_feats[name_attr]['amount_control'] += 1
                        icpc_feats[name_attr]['amount'] += 1
                        icpc_feats[name_attr]['possible_amount_control'] +=1
                        total_icpc_per_patient +=1
                        total_icpc_per_patient_control +=1
                        amount_icpcs_control +=1
                        amount_icpcs +=1
                        index+=1
                        continue
                    
                    else:
                        icpc_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            icpc_feats[name_attr]['amount_control'] += 1
                            icpc_feats[name_attr]['amount'] += 1
                            icpc_feats[name_attr]['possible_amount_control'] +=1
                            total_icpc_per_patient +=1
                            total_icpc_per_patient_control +=1
                            amount_icpcs_control +=1
                            amount_icpcs +=1
                            index+=1
                            continue
                    else:
                        icpc_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

            
            if name_attr.split('_')[1] in ['ATC', 'atc']:
                if not name_attr in atc_feats:
                    set_atcs.append(name_attr)
                    set_atcs_control.append(name_attr)
                    atc_feats[name_attr]={'amount_stroke': 0, 'amount_control':0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    
                    if attr >=1:
                        atc_feats[name_attr]['amount_control'] += 1
                        atc_feats[name_attr]['amount'] += 1
                        atc_feats[name_attr]['possible_amount_control'] +=1
                        total_atc_per_patient +=1
                        total_atc_per_patient_control +=1
                        amount_atcs_control +=1
                        amount_atcs +=1
                        index+=1
                        continue

                    else:
                        atc_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            atc_feats[name_attr]['amount_control'] += 1
                            atc_feats[name_attr]['amount'] += 1
                            atc_feats[name_attr]['possible_amount_control'] +=1
                            total_atc_per_patient +=1
                            total_atc_per_patient_control +=1
                            amount_atcs_control +=1
                            amount_atcs +=1
                            index+=1
                            continue
                    else:
                        atc_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

            if name_attr.split('_')[0] in ['i', 'd', 's', 'h', 'l', 'n', 'N','L','H','S','D','I']:
                if not name_attr in lab_feats:
                    set_lab.append(name_attr)
                    set_lab_control.append(name_attr)
                    lab_feats[name_attr]={ 'amount_stroke':0, 'amount_control': 0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}

                    if attr >=1:
                        lab_feats[name_attr]['amount_control'] += 1
                        lab_feats[name_attr]['amount'] += 1
                        lab_feats[name_attr]['possible_amount_control'] +=1
                        total_lab_per_patient +=1
                        total_lab_per_patient_control +=1
                        amount_lab_control +=1
                        amount_lab +=1
                        index+=1
                        continue

                    else:
                        lab_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1:
                            lab_feats[name_attr]['amount_control'] += 1
                            lab_feats[name_attr]['amount'] += 1
                            lab_feats[name_attr]['possible_amount_control'] +=1
                            total_lab_per_patient +=1
                            total_lab_per_patient_control +=1
                            amount_lab_control +=1
                            amount_lab +=1
                            index+=1
                            continue
                    else:
                        lab_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

            else:
                if not name_attr in other_feats:
                    set_other.append(name_attr)
                    set_other_control.append(name_attr)
                    other_feats[name_attr]= { 'amount_stroke':0, 'amount_control': 0, 'amount':0, 'possible_amount_control':0, 'possible_amount_stroke':0}
                    # other_feats[name_attr]['amount'] = 0 #{'amount': 0}

                    if attr >=1 :
                        other_feats[name_attr]['amount_control'] += 1
                        other_feats[name_attr]['amount'] += 1
                        other_feats[name_attr]['possible_amount_control'] +=1
                        total_other_per_patient+=1
                        total_other_per_patient_control+=1
                        index+=1
                        continue
                    else:
                        other_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue

                else:
                    if attr >=1 :
                        other_feats[name_attr]['amount_control'] += 1
                        other_feats[name_attr]['amount'] += 1
                        other_feats[name_attr]['possible_amount_control'] +=1
                        total_other_per_patient+=1
                        total_other_per_patient_control+=1
                        index+=1
                        continue
                    else:
                        other_feats[name_attr]['possible_amount_control'] +=1
                        index+=1
                        continue


    mean_icpcs_per_patient.append(total_icpc_per_patient)
    mean_icpcs_per_patient_stroke.append(total_icpc_per_patient_stroke)
    mean_icpcs_per_patient_control.append(total_icpc_per_patient_control)

    mean_atcs_per_patient.append(total_atc_per_patient)
    mean_atcs_per_patient_stroke.append(total_atc_per_patient_stroke)
    mean_atcs_per_patient_control.append(total_atc_per_patient_control)

    mean_other_per_patient.append(total_other_per_patient)
    mean_other_per_patient_stroke.append(total_other_per_patient_stroke)
    mean_other_per_patient_control.append(total_other_per_patient_control)

    mean_lab_per_patient.append(total_lab_per_patient)
    mean_lab_per_patient_stroke.append(total_lab_per_patient_stroke)
    mean_lab_per_patient_control.append(total_lab_per_patient_control)


print(t1,t2)
final_dict = {}

mean_icpcs_per_patient = np.mean(mean_icpcs_per_patient)
mean_icpcs_per_patient_stroke = np.mean(mean_icpcs_per_patient_stroke)
mean_icpcs_per_patient_control = np.mean(mean_icpcs_per_patient_control)

mean_atcs_per_patient = np.mean(mean_atcs_per_patient)
mean_atcs_per_patient_stroke = np.mean(mean_atcs_per_patient_stroke)
mean_atcs_per_patient_control = np.mean(mean_atcs_per_patient_control)

mean_other_per_patient = np.mean(mean_other_per_patient)
mean_other_per_patient_stroke = np.mean(mean_other_per_patient_stroke)
mean_other_per_patient_control = np.mean(mean_other_per_patient_control)

mean_lab_per_patient = np.mean(mean_lab_per_patient)
mean_lab_per_patient_stroke = np.mean(mean_lab_per_patient_stroke)
mean_lab_per_patient_control = np.mean(mean_lab_per_patient_control)

mean_age_control = np.mean(mean_age_control)
mean_age_stroke = np.mean(mean_age_stroke)
mean_age = np.mean(mean_age)

unique_icpcs_stroke = len(set(set_icpcs_stroke))
unique_icpcs_control = len(set(set_icpcs_control))
unique_icpcs = len(set(set_icpcs))

unique_atcs_stroke = len(set(set_atcs_stroke))
unique_atcs_control = len(set(set_atcs_control))
unique_atcs = len(set(set_atcs))

unique_lab_stroke = len(set(set_lab_stroke))
unique_lab_control = len(set(set_lab_control))
unique_lab = len(set(set_lab))

unique_other_stroke = len(set(set_other_stroke))
unique_other_control = len(set(set_other_control))
unique_other = len(set(set_other))

final_dict['mean_age_control'] = mean_age_control
final_dict['mean_age_stroke'] = mean_age_stroke
final_dict['mean_age'] = mean_age

final_dict['amount_men_stroke'] = amount_men_stroke
final_dict['amount_women_stroke'] = amount_women_stroke
final_dict['amount_men'] = amount_men
final_dict['amount_women'] = amount_women

final_dict['total_stroke'] = total_stroke
final_dict['total_control'] = total_control

final_dict['unique_icpcs_stroke'] = unique_icpcs_stroke
final_dict['unique_icpcs_control'] = unique_icpcs_control
final_dict['unique_icpcs'] = unique_icpcs
final_dict['amount_icpcs_stroke'] = amount_icpcs_stroke
final_dict['amount_icpcs_control'] = amount_icpcs_control
final_dict['amount_icpcs'] = amount_icpcs
final_dict['mean_icpc_per_patient_stroke'] = mean_icpcs_per_patient_stroke
final_dict['mean_icpc_per_patient_control'] = mean_icpcs_per_patient_control
final_dict['mean_icpc_per_patient'] = mean_icpcs_per_patient

final_dict['unique_atcs_stroke'] = unique_atcs_stroke
final_dict['unique_atcs_control'] = unique_atcs_control
final_dict['unique_atcs'] = unique_atcs
final_dict['amount_atcs_stroke'] = amount_atcs_stroke
final_dict['amount_atcs_control'] = amount_atcs_control
final_dict['amount_atcs'] = amount_atcs
final_dict['mean_atc_per_patient_stroke'] = mean_atcs_per_patient_stroke
final_dict['mean_atc_per_patient_control'] = mean_atcs_per_patient_control
final_dict['mean_atc_per_patient'] = mean_atcs_per_patient

final_dict['unique_lab_stroke'] = unique_lab_stroke
final_dict['unique_lab_control'] = unique_lab_control
final_dict['unique_lab'] = unique_lab
final_dict['amount_lab_stroke'] = amount_lab_stroke
final_dict['amount_lab_control'] = amount_lab_control
final_dict['amount_lab'] = amount_lab
final_dict['mean_lab_per_patient_stroke'] = mean_lab_per_patient_stroke
final_dict['mean_lab_per_patient_control'] = mean_lab_per_patient_control
final_dict['mean_lab_per_patient'] = mean_lab_per_patient

final_dict['unique_other_stroke'] = unique_other_stroke
final_dict['unique_other_control'] = unique_other_control
final_dict['unique_other'] = unique_other
final_dict['amount_other_stroke'] = amount_other_stroke
final_dict['amount_other_control'] = amount_other_control
final_dict['amount_other'] = amount_other
final_dict['mean_other_per_patient_stroke'] = mean_other_per_patient_stroke
final_dict['mean_other_per_patient_control'] = mean_other_per_patient_control
final_dict['mean_other_per_patient'] = mean_other_per_patient

final_dict['total_stroke'] = total_stroke
final_dict['total_control'] = total_control

percentage_dict = {}
percentage_dict['meaning of percentages'] = ['perc of feature incidence of stroke group in total cohort' ,
                                             'perc of feature incidence of control group in total cohort',
                                             'perc of actual feature incidence of total possible incidence within stroke group',
                                             'perc of actual feature incidence of total possible incidence within control group']

for key,value in other_feats.items():
    amt_stroke = other_feats[key]['amount_stroke']
    amt_control = other_feats[key]['amount_control']
    pos_amt_stroke = other_feats[key]['possible_amount_stroke']
    pos_amt_control = other_feats[key]['possible_amount_control']
    amt = other_feats[key]['amount']

    try:
        percentage_dict[key] = [round(((amt_stroke/amt) * 100), 2), round(((amt_control/amt) * 100), 2), round(((amt_stroke/pos_amt_stroke) * 100), 2), round(((amt_control/pos_amt_control) * 100), 2)]
    except ZeroDivisionError:
        percentage_dict[key] = 'unknown'

for key,value in lab_feats.items():
    amt_stroke = lab_feats[key]['amount_stroke']
    amt_control = lab_feats[key]['amount_control']
    pos_amt_stroke = lab_feats[key]['possible_amount_stroke']
    pos_amt_control = lab_feats[key]['possible_amount_control']
    amt = lab_feats[key]['amount']

    try:
        percentage_dict[key] = [round(((amt_stroke/amt) * 100), 2), round(((amt_control/amt) * 100), 2), round(((amt_stroke/pos_amt_stroke) * 100), 2), round(((amt_control/pos_amt_control) * 100), 2)]
    except ZeroDivisionError:
        percentage_dict[key] = 'unknown'
        
for key,value in icpc_feats.items():
    amt_stroke = icpc_feats[key]['amount_stroke']
    amt_control = icpc_feats[key]['amount_control']
    pos_amt_stroke = icpc_feats[key]['possible_amount_stroke']
    pos_amt_control = icpc_feats[key]['possible_amount_control']
    amt = icpc_feats[key]['amount']

    try:
        percentage_dict[key] = [round(((amt_stroke/amt) * 100), 2), round(((amt_control/amt) * 100), 2), round(((amt_stroke/pos_amt_stroke) * 100), 2), round(((amt_control/pos_amt_control) * 100), 2)]
    except ZeroDivisionError:
        percentage_dict[key] = 'unknown'

for key,value in atc_feats.items():
    amt_stroke = atc_feats[key]['amount_stroke']
    amt_control = atc_feats[key]['amount_control']
    pos_amt_stroke = atc_feats[key]['possible_amount_stroke']
    pos_amt_control = atc_feats[key]['possible_amount_control']
    amt = atc_feats[key]['amount']

    try:
        percentage_dict[key] = [round(((amt_stroke/amt) * 100), 2), round(((amt_control/amt) * 100), 2), round(((amt_stroke/pos_amt_stroke) * 100), 2), round(((amt_control/pos_amt_control) * 100), 2)]
    except ZeroDivisionError:
        percentage_dict[key] = 'unknown'

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/statistics.txt'
with open(f_out, 'w') as f:
    		 for key, value in final_dict.items():
        		f.write('%s:%s\n' % (key, value))

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/other_counts.txt'
with open(f_out, 'w') as f:
    		 for key, value in other_feats.items():
        		f.write('%s:%s\n' % (key, value))

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/lab_counts.txt'
with open(f_out, 'w') as f:
    		 for key, value in lab_feats.items():
        		f.write('%s:%s\n' % (key, value))

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/icpc_counts.txt'
with open(f_out, 'w') as f:
    		 for key, value in atc_feats.items():
        		f.write('%s:%s\n' % (key, value))

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/icpc_counts.txt'
with open(f_out, 'w') as f:
    		 for key, value in icpc_feats.items():
        		f.write('%s:%s\n' % (key, value))

f_out='/Users/Tristan/Downloads/FINAL_RESULTS/statistics/percentages_stroke_control.txt'
with open(f_out, 'w') as f:
    		 for key, value in percentage_dict.items():
        		f.write('%s:%s\n' % (key, value))


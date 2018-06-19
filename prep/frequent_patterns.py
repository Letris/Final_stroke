import sys
from collections import defaultdict
import numpy as np
from operator import itemgetter

import util_.in_out as in_out
from prep.Pattern import Pattern, enrich_pattern
from prep.StateInterval import StateInterval
from prep.StateSequence import StateSequence

from prep.EnrichProcesses import fill_enrichment_dicts
from tqdm import *
from collections import defaultdict

global ENRICHMENT_DICT

# mine all frequent patterns (> min_sup) in the specified file
def generate(sequence_file, min_sup, mapping_dir, verbose=False):
	# global ENRICHMENT_DICT
	# ENRICHMENT_DICT = import_enrichment_dicts(mapping_dir)

	if type(sequence_file) == dict:
		# print(sequence_file)
		# sequences = (v['data'] for k, v in sequence_file.items())
		sequences = [v['data'] for k, v in sequence_file.items()]

	else:
		sequences = in_out.read_csv(sequence_file)

	sequences_pos = patients_with_class_val(sequences, ['positive', True])

	if bool(sequences_pos) == False:
		print('No positive examples available in this subset of the data')

	if type(sequence_file) == dict:
		sequences = [v['data'] for k, v in sequence_file.items()]
		# sequences = (v['data'] for k, v in sequence_file.items())
	else:
		sequences = in_out.read_csv(sequence_file)

	sequences_neg = patients_with_class_val(sequences, ['negative', False])

	if verbose: 
		print ("Patient dict:")
		print (sequences_neg)

	print ('###### Mining positive stroke freq patterns ######')
	frequent_patterns_pos, MPTP_pos = mine(sequences_pos, min_sup, sequences_neg, verbose)
	print ('###### Mining negative stroke freq patterns ######')
	frequent_patterns_neg, MPTP_neg = mine(sequences_neg, min_sup, sequences_pos, verbose)
	
	# print frequent_patterns_pos
	# print frequent_patterns_neg
	# save for later use !
	# MPTP_pos.update(MPTP_neg)
	# print len(list(set(frequent_patterns_pos+frequent_patterns_neg)))
	# print len(MPTP_pos)
	print ('###### Done mining patterns ######')
	return list(set(frequent_patterns_pos+frequent_patterns_neg)), frequent_patterns_pos, frequent_patterns_neg

def patients_with_class_val(arr, class_val):
	result = {}

	if isinstance(arr[0][-1], list):
		for lst in arr:
			if lst[-1][0] in class_val:
				result[lst[0]] = lst[1:-1]
	else:
		for lst in arr:
			if lst[-1] in class_val:
				result[lst[0]] = lst[1:-1]

	for k in result:
		record = result[k][2:]
		gender = result[k][1]
		birthyear = result[k][0]
		for i in range(len(record)):
			if type(record[i]) != StateInterval:
				str_SI = record[i].split(';')
				record[i] = StateInterval(str_SI[0], str_SI[1], str_SI[2])
		result[k] = StateSequence(record, k, gender, birthyear)
	
	return result

# import cProfile, pstats
# Fx means Frequent x-patterns
def mine(data_dict, min_sup, negated_data, verbose):
	print("...generating 1-patterns")
	
	F1 = generate_F1(data_dict, min_sup, negated_data)
	if verbose: print (F1)
	k_max = len(max(data_dict.values(), key=len)) if len(data_dict)!=0 else 0 # k = length of the longest instance, because that is how the maximum number of combinations possible for this specific group
	if verbose: print ("longest patient sequence: " + str(k_max))
	
	F1 = set(F1.values()) #values
	new_F1 = F1
	Fk = set(F1)
	
	MPTPs = set
	for p in F1:
		if p.is_significant(alpha=0.05):
				MPTPs.add(p)

	# MPTPs = set([p for p in F1 if p.is_significant(alpha=0.05)])
	candidates = list(F1)
	print ("...found " + str(len(candidates)))
	result = list(candidates)
	
	for k in range(0,k_max):
		print("...generating " + str(k+2) + '-patterns'),

		potential_Fk, new_F1 = generate_candidates(candidates, Fk, new_F1, min_sup, data_dict)
		Fk = evaluate_support(potential_Fk, data_dict, min_sup, negated_data, MPTPs)
		print('usha')
		print(Fk)
		result = result + Fk
		print ("...found " + str(len(Fk)))
		if len(Fk) == 0:
			break
		if k==1: 
			break
	return result, MPTPs#[c for c in result if c.is_frequent]

def generate_F1(data_dict, min_sup, negated_data):
	len_data = len(data_dict)

	instances = [data_dict[sequence].get_states() for sequence in data_dict ]
	y_occurrences = defaultdict(int)
	for states in instances:
		for state in set(states): # for all unique states
			y_occurrences[state] += 1

	negated_instances = [negated_data[sequence].get_states() for sequence in negated_data ]
	all_occurrences = dict(y_occurrences)
	for states in negated_instances:
		for state in set(states): # for all unique states
			if state not in all_occurrences:
				all_occurrences[state] = 0
			all_occurrences[state] += 1

	# calc frequent patterns (more freq than min_sup)
	init_mcs = len_data / len_data + len(negated_data)
	# init_mc = len(data_dict) / (len(data_dict) + len(negated_data))

	F1=dict()
	for key in y_occurrences:
		states = np.array([key])
		relations = np.empty((1,1), dtype='S1')
		relations.fill(' ')
		loc_support = y_occurrences[key]/float(len_data)
		y_support = y_occurrences[key]/float(len_data+len(negated_data))
		all_support = all_occurrences[key]/float(len_data+len(negated_data))
		if loc_support >= min_sup:
			F1[key] = Pattern(states, relations, sup=loc_support, subpatterns=[], in_candidates=True)
			F1[key].generate_id_list(data_dict, False)
			F1[key].generate_id_list(negated_data, True)
			F1[key].is_frequent = (loc_support >= min_sup)
			F1[key].mcs = init_mcs
			F1[key].confidence = y_support/float(all_support)
			F1[key].calc_max_conf()
	return F1

# for pseudocode, see "A Temporal Pattern Mining Approach for Classifying Electronics Health Record Data"
def generate_candidates(all_patterns, Fk, F1, min_sup, data_dict):

	# Fk_singletons = set()
	# for fp1 in F1:
	# 	for fpk in Fk:
	# 		for state in fpk.states:
	# 			if state == fp1.states[0]:
	# 				Fk_singletons.add(fp1)
	Fk_singletons = set(fp1 for fp1 in F1 for fpk in Fk for state in fpk.states if state == fp1.states[0])
	
	len_data = float((len(data_dict)))

	candidates = []
	for k_pattern in tqdm(Fk): # for all frequent k-patterns
		for singleton in tqdm(Fk_singletons): # for all unique singleton states in the current Fk states
			superpatterns, all_patterns = generate_coherent_superpatterns(k_pattern, singleton, data_dict, all_patterns) # generate new candidates

			for superpattern in tqdm(superpatterns): # for each k+1-pattern created, we check if it is frequent
				
				subpatterns = superpattern.get_subpatterns()
				
				# intersection of all created subpatterns and all frequent k-patterns 
				frequent_subs = [pattern for pattern in Fk if pattern in subpatterns]
				
				# if the created subpatterns are all in Fk
				if len(frequent_subs) >= len(subpatterns):
					if len(frequent_subs) > len(subpatterns):
						print ('> (impossible)')
					superpattern.generate_pid_list(frequent_subs)
					# superpattern.generate_pid_list(frequent_subs, negated=True)
					# superpattern.calc_max_conf_subpatterns(frequent_subs)
					

					if (len(superpattern.pid_list)/len_data) > min_sup:
						candidates.append(superpattern)
	return candidates, Fk_singletons

def evaluate_support(patterns, data, min_sup, negated_data, MPTPs):
	len_data = len(data)

	result = []
	p_values = []
	for p in patterns:
		p.generate_id_list(data, negated=False)
		p.calculate_support(min_sup, len_data)

		if p.support >= min_sup:
			result.append(p)
		# else:
		# 	print(p.support)
		# 	print('seriously')

	return result

def generate_coherent_superpatterns(P, S, data_dict, all_patterns, strict_matching=True):

	superpatterns = []
	rel_length = P.relations.shape[1] + 1
	rels = np.empty((rel_length), dtype='S1')
	rels.fill('b')
	rels[0] = ' '

	new_p = enrich_pattern(P, S, rels, data_dict)
	if new_p not in all_patterns:
		superpatterns.append(new_p)
		all_patterns.append(new_p)

	if strict_matching:
		for i in range(1, len(rels)):

			# semantic abstraction compatibility check
			# if not compatible, adding another 'c' will never make it compatible
			# so we safely return!
			if not compatible(P.states[i-1], S.states[0]): 
				return superpatterns, all_patterns

			rels[i] = 'c'
			new_p = enrich_pattern(P, S, rels, data_dict)
			if new_p not in all_patterns:
				superpatterns.append(new_p)
				all_patterns.append(new_p)

	return superpatterns, all_patterns

def compatible(p_state, s_state):
	# global ENRICHMENT_DICT

	# get enrichment type (e.g. effect, association)
	new_state = s_state.split('_')[0]
	new_state_type = s_state.split('_')[-1]

	other_state = p_state.split('_')[0]
	other_state_type = p_state.split('_')[-1]

	if new_state_type in ['icpc', 'association', 'manifestationof'] and not (other_state_type in ['atc', 'effect', 'indication', 'ingredient']):
		src = 'icpc'
	elif new_state_type in ['atc', 'effect', 'indication', 'ingredient'] and not (other_state_type in ['icpc', 'association', 'manifestationof']):
		src = 'atc'
	else: # no semantic abstraction was done..
		return True

	if new_state_type == src and other_state_type == src:
		return True # two of the same codes, either icpc or atc, we dont care

	# possibly_compatible_list_keys = ENRICHMENT_DICT[src]
	# for lst in possibly_compatible_list_keys.values():
	# 	compat = not (new_state in lst and other_state in lst)
	# 	if not compat:
	# 		# print '***'
	# 		# print new_state, new_state_type
	# 		# print other_state, other_state_type
	# 		# print len(lst), lst
	# 		return False

	# if new_state_type in ['atc', 'effect', 'indication', 'ingredient']:
	# 	for _type in other_state_types:
	# 		if new_state_type == 'atc' and other_state_type == 'atc':
	# 			continue # two atc codes, we dont care

	# 		possibly_compatible_list_keys = ENRICHMENT_DICT['atc']
	# 		for key, lst in possibly_compatible_list_keys.items():
	# 			lst = [key] + lst
	# 			b = new_state in lst and any(state in lst for state in other_states)
	# 			print len(lst), b
	# 			if b: 
	# 				return False

	return True # does not occur in the same list, so it is compatible

def import_enrichment_dicts(mapping_dir):
	dct = fill_enrichment_dicts(mapping_dir)

	d1 = stem_atc(dct['effects'])
	d2 = stem_atc(dct['indications'])
	d3 = stem_atc(dct['ingredients'])
	dct['atc'] = { key: ([key] + d1.get(key, []) + d2.get(key, []) + d3.get(key, [])) for key in set( d1.keys() + d2.keys() + d3.keys()) }

	d1 = dct['association']
	d2 = dct['manifestationof']
	dct['icpc'] = { key: ([key] + d1.get(key, []) + d2.get(key, [])) for key in set( d1.keys() + d2.keys() ) }

	del dct['effects']
	del dct['indications']
	del dct['ingredients']
	del dct['association']
	del dct['manifestationof']

	return dct

def stem_atc(d):
	'''stems the atc codes (the keys) in dict d to the first three characters'''
	d2 = defaultdict(list)
	for key in d:
		d2[key[0:3]] = d2[key[0:3]] + d[key]
	return d2

if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) != 1 and len(args) != 2:
		print ("faulty input argument(s). try again")
		print ("usage: python frequent_patterns.py path/to/state_sequences.py")
		print ("and optinally, add a minimum support (default 0.1)")
		exit()
	if len(args) == 2:
		generate(args[0], min_sup=float(args[1]), verbose=False)
	elif len(args) == 1:
		generate(args[0], verbose=False)

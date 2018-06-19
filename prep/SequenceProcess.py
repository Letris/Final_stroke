from operator import attrgetter
import re

from prep.PreProcess import PreProcess
from prep.StateInterval import StateInterval
from prep.date_math import get_dates, str2date, four_weeks
import util_.util as util
from prep.abstracts import get_value, get_trends
from collections import defaultdict
from tqdm import *
from util_.in_out import write_csv, save_obj, load_obj
import pandas as pd
import numpy as np

class SequenceProcess(PreProcess):
	'''class describing sequential/temporal way of preprocessing 
		the data by analyzing patterns of data concepts'''

	def insert_data(self, rows, headers, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False, counter=0):
		'''inserts data from the specified csv and corresponding columns'''
	
		important_features = ['CHOLBMT', 'RRDIKA', 'RRSYKA']

		# read rows into list to re-use
		rows = rows.where((pd.notnull(rows)), None)

		# make convenient reference to the dictionary
		dct = self.id2data

		# # get data and corresponding headers
		# rows, headers = util.import_data(f, delim=self.delim)

		# get the index of the relevant columns
		# ID_idx = headers.index(self.ID_column)
		code_idx = headers.index(code_column) + 1
		date_idx = headers.index(date_column[0]) + 1
		b_date_idx = headers.index(date_column[0]) + 1
		e_date_idx = headers.index(date_column[1]) + 1
		
		# if incorporate_SOEP:
		# 	SOEP_idx = headers.index(incorporate_SOEP)
		
		# regex pattern to match (ATC/ICPC standards)
		pattern = re.compile(regex_string)

		# regex pattern to match (ATC/ICPC standards)
		pattern = re.compile(regex_string)

		if 'lab_results' in suffix:
			values_dict = dict()
			# val_idx = headers.index('valuen') + 1

		# pair IDs with a dict corresponding to data and dates
			for row in rows.itertuples():#line in de data
				code = row[code_idx]
				# if we do not know the high and low values, determine by data distribution
				if code not in important_features:
					if not code in values_dict:
						try:
							values_dict[code] = [float(row.valuen)]
						except ValueError:
							continue
						except TypeError:
							continue
					else:
						try:
							values_dict[code].append(float(row.valuen))
						except ValueError:
							continue
						except TypeError:
							continue
							
			minmax_dict = self.calculate_minmax(values_dict, pattern, limit)
		

		# keep track of number of times the row is attributed to a positive stroke patient (or patient where the target instance = 'positive')
		num_pos = 0
		num_total = 0
		attribute_count = dict()
		# iterate over all instances, making a new dict with the new attributes as keys
		attribute2ids = dict()

		max=100000000000000000
		current = 0 

		# iterate over all instances
		for row in tqdm(rows.itertuples()):
			current += 1	
			# row = row.split(';')

			if current > max: 
				break
			else:
				num_total+=1

				# if key is not in the data dictionary, we skip it
				key = row.Index
				
				if not key in dct:
					continue

			# init other vars
			b_date = str2date(row[b_date_idx], give_default_begin=True) # begin of event
			e_date = str2date(row[e_date_idx], give_default_end=True) # end of event
			b_reg = dct[key]['stroke_dates'][1] # beginning of registration
			e_reg = dct[key]['stroke_dates'][2] # ending of registration
			# print('wddup')
			# print(b_reg, e_reg)
			# print('xxx')

			# print(dct[key]['stroke_dates'][3], dct[key]['stroke_dates'][4])
			original_code = row[code_idx]
			if original_code == None:
				continue

			truncated_code = self.generate_code(original_code, limit) 
			if truncated_code == None or truncated_code in ['K90', 'K89', 'k90', 'k89']:
				continue

			print(b_reg, b_date, e_date)
			# print(b_reg <= b_date)
			# print(b_date <= e_reg)
			# print(b_reg <= e_date)
			# print(e_date <= e_reg)
			# if in the required interval (either beginning or ending date) AND code is valid
			if ( (b_reg <= b_date and b_date <= e_reg) or (b_reg <= e_date and e_date <= e_reg) ) and pattern.match(truncated_code):
				
				# if we need to take the SOEP code of consults into account
				# if (not incorporate_SOEP) or (incorporate_SOEP and row[SOEP_idx] == 'E'):

					# generate attribute names
					if 'lab_results' in suffix: # if we prepare for lab result abstraction
					
						try:
								val = float(row.valuen)
								if not original_code in important_features:
									min_val = minmax_dict[truncated_code]['low_bound']
									max_val = minmax_dict[truncated_code]['high_bound']

								else:
									min_val, max_val = self.determine_minmax(original_code)

		
						except ValueError:
								continue

						except TypeError:
									continue

						val, min_val, max_val = self.make_lab_values(val, min_val, max_val)

						if not 'ID2abstractions' in locals():
							# dict (patient) of dict (lab measurement name) of list of tuples (all value/date combinations of measurement)
							ID2abstractions = dict()
						
						util.init_key(ID2abstractions, key, dict())
						util.init_key(ID2abstractions[key], original_code, [])

						ID2abstractions[key][original_code].append((b_date, val))
					
						if '' not in [val, min_val, max_val]:
							attributes = [get_value(val, min_val, max_val, original_code)]

							# # add value abstraction as state interval
							# self.insert_state_interval(key, attr, b_date, e_date)
						else:
							attributes = []

					else:
						if 'cardiometabolism' in suffix:
							val_idx = headers.index('valuec')
							value = str(row[val_idx])
							
						else:
							value = None

						attributes = self.generate_attributes(original_code, limit, suffix, value, src=code_column)

					# this loop allows multiple attributes to be created in the previous code line
					# this allows for other classes to subclass this class, e.g. SequenceEnrichProcess
					for attr in attributes:
						if 'allergies' in suffix:
									# val_idx = headers.index('flag')
									value = row.flag

									# check if the person actually has the allergie for which was tested
									if value == 'POS':
										self.insert_state_interval(key, attr, b_date, e_date, original_code, code_column)
									# if negative or not tested, it is assumed that person does not have particular allergie
									else:
										continue
						# insert a StateInterval object with the specified parameters
						self.insert_state_interval(key, attr, b_date, e_date, original_code, code_column)


		if suffix == 'lab_results': # do funky stuff with trends and abstractions
			# convert to trends PER lab result
			for ID in ID2abstractions:
				# print ID2abstractions[ID]
				for k, points in ID2abstractions[ID].items():
					
					# the values are sorted before abstraction
					points = sorted(list(set(points)))

					# abstract the values and append to the current patient's sequence
					# if only 1 measurement was done, we cannot do time series analysis
					if len(points) > 1 and ID in dct: 
						abstractions = get_trends(k, points)
						for abstraction in abstractions:
							self.insert_state_interval(ID, *abstraction, original_code=original_code, src=code_column)
						# self.id2data[ID]['data'] = self.id2data[ID]['data'] + abstractions
		
		# add data to each instance
		to_save = {}

		for ID in dct:
			to_save[ID] = []

		for ID in dct:
			data = dct[ID]['data']
			# to_save[ID] = []

			for id2occurrences in attribute2ids.values():
				
				# if patient has occurrences for the attribute, add that number, else add 0
				if ID in id2occurrences: 
					data.append(id2occurrences[ID])
					to_save[ID].append(id2occurrences[ID])

				else:
					data.append(0)
					to_save[ID].append(0)

		if self.survival == True:
			save_obj(to_save, self.in_dir + suffix[0] + '_dict_temporal' + str(counter)+ '_survival' + '.pkl')
			save_obj(list(attribute2ids.keys()), self.in_dir + suffix[0]  + 'temporal_headers'+ str(counter) + '.pkl')
		else:
			save_obj(to_save, self.in_dir + suffix[0] + '_dict_temporal' + str(counter) + '.pkl')
			save_obj(list(attribute2ids.keys()), self.in_dir + suffix[0] + 'temporal_headers'+  str(counter) + '.pkl')
		# to satisfy return value requirement for the method 'process' in the superclass
		return [], -1, -1
			
	def insert_state_interval(self, key, state, begin, end, original_code, src):
		'''converts state-begin-end-triples to state intervals, add to data record data'''
		sequence = self.id2data[key]['data']
		SI = StateInterval(state, begin, end)
		sequence.append(SI)

	def sort_sequences(self):
	 	# '''sort each state sequence (= 1 per patient) consisting of state intervals
		#  	order of sort is: begin date->end date->lexical order of state name'''
		for k in self.id2data:
			sequence = self.id2data[k]['data']
			static_seq = sequence[0:3] # gender/age
			dynamic_seq = sequence[5:-1]
			dynamic_seq.sort(key=attrgetter('begin', 'end', 'state'))
			stroke = [sequence[-1]]
			self.id2data[k]['data'] = static_seq + dynamic_seq + stroke
	
	def generate_code(self, code, limit):
		'''generates the required part of the code in a field, 
			e.g. atc code A01 in field A01B234'''
		try:
			result = code.upper().strip()[0:limit]
		except AttributeError:
			return

		if result.lower() in ['oncologie', 'chirurgie', 'gastro-enterologie', 'interne geneeskunde', 'scopie-afdeling']:
			result = None
		#'interne geneeskunde           ','gastro-enterologie            ', 
		return result

	def calculate_minmax(self, values_dict, pattern, limit):
		''' if high/low values are unknown '''

		minmax_dict = dict()
		
		for code, values in values_dict.items():
			
			original_code = code
			if original_code == None:
				continue

			truncated_code = self.generate_code(original_code, limit)
			if truncated_code == None:
				continue

			

			if pattern.match(truncated_code):
				value_list = np.array(values)
		# contextualize based on data distribution
		
				low_bound = np.percentile(value_list, 25)
				high_bound = np.percentile(value_list, 75)
				minmax_dict[truncated_code] = {'low_bound' : low_bound, 'high_bound' : high_bound}
	
		return minmax_dict

	def determine_minmax(self, code):
		''' based on known high/low values '''

		if code == 'RRDIKA':
			return 60, 90
		if code == 'RRSYKA': 
			return 90, 140
		if code == 'CHOLBMT':
			return 5, 6.5
	
if __name__ == '__main__':
	import sys

	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	age_range = (30,150)

	seq_p = SequenceProcess()
	seq_p.process(in_dir, out_dir, age_range)
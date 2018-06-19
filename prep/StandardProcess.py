import re

from prep.PreProcess import PreProcess
from prep.date_math import str2date, four_weeks
import util_.util as util
from prep.abstracts import get_value, get_trends
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import *
from numba import jit
from util_.in_out import write_csv, save_obj, load_obj

class StandardProcess(PreProcess):
	'''class describing standard way of preprocessing 
		the data by counting occurrences of data concepts'''

	def insert_data(self, rows, headers, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False, counter=0):
		'''inserts data from the specified csv and corresponding columns'''
	
		important_features = ['CHOLBMT', 'RRDIKA', 'RRSYKA']

		# make convenient reference to the dictionary
		dct = self.id2data
		rows = rows.where((pd.notnull(rows)), None)

		# # get data and corresponding headers
		# rows, headers = util.import_data(f, delim=self.delim)

		# get the index of the relevant columns
		# ID_idx = headers.index(self.ID_column)
		code_idx = headers.index(code_column) + 1
		date_idx = headers.index(date_column[0]) + 1
		
		
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
			

		if incorporate_SOEP:
			SOEP_idx = headers.index(incorporate_SOEP)

		# keep track of number of times the row is attributed to a positive stroke patient (or patient where the target instance = 'positive')
		num_pos = 0
		num_total = 0
		attribute_count = dict()
		# iterate over all instances, making a new dict with the new attributes as keys
		attribute2ids = dict()

		max=10000000000000000000
		current = 0 

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

				if dct[key]['stroke_dates'][0] != 'negative':
					num_pos+=1

				# init other vars
				date = str2date(row[date_idx], give_default_begin=True, give_default_end=True)
				begin = dct[key]['stroke_dates'][1]
				end = dct[key]['stroke_dates'][2]

				if code_column == 'specialisme':
					end = end - four_weeks()

				original_code = row[code_idx]
				if original_code == None:
					continue

				truncated_code = self.generate_code(original_code, limit)
				if truncated_code == None or truncated_code in ['K90', 'K89', 'k90', 'k89']:
					continue
				
				
				# if in the required interval and code is valid
				if (begin <= date and date <= end) and pattern.match(truncated_code):
					# if we do not care about SOEPcode (always except for journaal case) or the SOEPcode is E
					# if (not incorporate_SOEP) or (incorporate_SOEP and row[SOEP_idx] == 'E'):
					
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

							if not 'ID2abstractions' in locals():
								# dict (patient) of dict (lab measurement name) of list of tuples (all value/date combinations of measurement)
								ID2abstractions = dict()
							
							util.init_key(ID2abstractions, key, dict())
							util.init_key(ID2abstractions[key], original_code, [])

							ID2abstractions[key][original_code].append((date, val))

							if '' not in [val, min_val, max_val]:
								attr = get_value(val, min_val, max_val, original_code)

								if not attr in attribute_count:
									attribute_count[attr] = 0

								# check if attribute name and ID instance already exist, if not, make them
								util.init_key(attribute2ids, attr, dict())
								util.init_key(attribute2ids[attr], key, 0)
								
								# add 1 to the occurrence of the attribute in the instance
								attribute2ids[attr][key] += 1
								attribute_count[attr] += 1

						else: # else no lab result collection, regular aggregation
							# generate attribute names

							if 'cardiometabolism' in suffix:
								# val_idx = headers.index('valuec')
								value = str(row.valuec)
							
							else:
								value = None

							attributes = self.generate_attributes(original_code, limit, suffix, value, src=code_column)
							# this loop allows multiple attributes to be created in the previous code line
							# this allows for other classes to subclass this class, e.g. StandardEnrichProcess
							for attr in attributes:
								if not attr in attribute_count:
									attribute_count[attr] = 0

								# print truncated_code, attr
								# check if attribute name and ID instance already exist, if not, make them
								util.init_key(attribute2ids, attr, dict())
								util.init_key(attribute2ids[attr], key, 0)

								# add 1 to the occurrence of the attribute in the instance, except if attribute is binary
								if 'smoking' in suffix:
									if attribute2ids[attr][key] == 1:
										continue

								if 'allergies' in suffix:
									# val_idx = headers.index('flag')
									value = row.flag

									# check if the person actually has the allergie for which was tested
									if value == 'POS':
										attribute2ids[attr][key] = 1
									# if negative or not tested, it is assumed that person does not have particular allergie
									else:
										attribute2ids[attr][key] = 0

								else:
								# if suffix[0] in ['atc', 'consults', 'actions', 'icpc', 'bmi', 'blood_pressure', 'alcohol', 'renal_function', 'lab_blood', 'lung_function']:
									attribute2ids[attr][key] += 1
									attribute_count[attr] += 1
		
		for attr, count in attribute_count.items():
			try:
				self.statistics[attr + '_count/min/max'] = [count, min_val, max_val]
			except UnboundLocalError:
				self.statistics[attr + '_count'] = count

		if 'lab_results' in suffix: # do funky stuff with trends and abstractions
			# convert to trends PER lab result
			for ID in ID2abstractions:
				# print ID2abstractions[ID]
				for k, points in ID2abstractions[ID].items():
					
					# the values are sorted before abstraction
					points = sorted(list(set(points)))

					# abstract the values and count the occurrences per measurement-trend per patient
					# if only 1 measurement was done, we cannot do time series analysis
					if len(points) > 1 and ID in dct: 
						abstractions = get_trends(k, points)
						for attr in abstractions:
							attr = attr[0] # get the state
							util.init_key(attribute2ids, attr, dict())
							util.init_key(attribute2ids[attr], ID, 0)
							attribute2ids[attr][ID] += 1
		# print len(attribute2ids)
		# print attribute2ids.keys()[0:5]
		
		

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

		save_obj(self.statistics, self.in_dir + suffix[0]+ '_statistics.pkl')

		if self.survival == True:
			save_obj(to_save, self.in_dir + suffix[0] + '_dict' + str(counter)+ '_survival' + '.pkl')
			save_obj(list(attribute2ids.keys()), self.in_dir + suffix[0]  + '_headers'+ str(counter) + '.pkl')
		else:
			save_obj(to_save, self.in_dir + suffix[0] + '_dict' + str(counter) + '.pkl')
			save_obj(list(attribute2ids.keys()), self.in_dir + suffix[0] + '_headers'+  str(counter) + '.pkl')


		# return the keys to be used as headers when writing the processed data
		return list(attribute2ids.keys()), num_total, num_pos, suffix

	def generate_code(self, code, limit):
		'''generates the required part of the code in a field, 
			e.g. atc code A01 in field A01B234'''
		if code == None: 
			code = ''

		try:
			result = code.upper().strip()[0:limit]
		except AttributeError:
			return

		if result.lower() in ['oncologie', 'chirurgie', 'gastro-enterologie', 'interne geneeskunde', 'scopie-afdeling']:
			result = None

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

	def mem_usage(self, pandas_obj):
		if isinstance(pandas_obj,pd.DataFrame):
			usage_b = pandas_obj.memory_usage(deep=True).sum()
		else: # we assume if not a df it's a series
			usage_b = pandas_obj.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
		return "{:03.2f} MB".format(usage_mb)

if __name__ == '__main__':
	dct = dict()
	dct['in_dir'] = '/Users/Reiny/Documents/UI_stroke/playground'
	dct['delimiter'] = ','
	dct['out_dir'] = '/Users/Reiny/Documents/UI_stroke/out'
	dct['min_age'] = 18
	dct['max_age'] = 150
	dct['begin_interval'] = int(365./52*38)
	dct['end_interval'] = int(365./52*12)
	dct['ID_column'] = 'patientnummer'

	sp = StandardProcess()
	sp.process(dct['in_dir'],
				dct['delimiter'],
				dct['out_dir'],
				dct['ID_column'],
				dct['min_age'],
				dct['max_age'],
				[int(dct['end_interval']), int(dct['begin_interval'])])
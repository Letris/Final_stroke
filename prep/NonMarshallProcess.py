import re

from prep.StandardProcess import StandardProcess
from prep.date_math import str2date, four_weeks
import util_.util as util
import prep.abstracts
from collections import defaultdict

class NonMarshallProcess(StandardProcess):
	'''class describing standard way of preprocessing 
		the data by counting occurrences of data concepts'''

	def insert_data(self, rows, headers, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False):
		'''inserts data from the specified csv and corresponding columns'''

			# read rows into list to re-use
		row_list = [row.split(';') for row in rows]

		# make convenient reference to the dictionary
		dct = self.id2data

		# # get data and corresponding headers
		# rows, headers = util.import_data(f, delim=self.delim)

		# get the index of the relevant columns
		ID_idx = headers.index(self.ID_column)
		code_idx = headers.index(code_column)
		date_idx = headers.index(date_column[0])
		
		if incorporate_SOEP:
			SOEP_idx = headers.index(incorporate_SOEP)
		
		# regex pattern to match (ATC/ICPC standards)
		pattern = re.compile(regex_string)

		if 'lab_results' in suffix:
			values_dict = defaultdict(dict) 
			val_idx = headers.index('valuen')

			for row in row_list:
				if not row[code_idx] in values_dict:
					try:
						values_dict[row[code_idx]] = [float(row[val_idx])]
					except ValueError:
						continue
				else:
					try:
						values_dict[row[code_idx]].append(float(row[val_idx]))
					except ValueError:
						continue
		
			
			minmax_dict = self.find_minmax(values_dict, pattern, limit)


		if incorporate_SOEP:
			SOEP_idx = headers.index(incorporate_SOEP)
		

		# make copy of rows to prevent emptying the original rows
		# if 'lab_results' in suffix:	
		# 	if 'alcohol' in suffix:
		# 		val_idx = headers.index('valuen')
		# 	else:
		# 		val_idx = headers.index('dvalue')

		# 	min_val, max_val = self.find_minmax(row_list, val_idx, code_idx, pattern, limit)

		# keep track of number of times the row is attributed to a positive stroke patient (or patient where the target instance = 'positive')
		num_pos = 0
		num_total = 0
		attribute_count = defaultdict(dict)
		# iterate over all instances, making a new dict with the new attributes as keys
		attribute2ids = defaultdict(dict)

		max=10000000000000000000
		current = 0 

		for row in row_list:
			# row = row.split(';')

			if current > max: 
				break
			else:
				num_total+=1

				# if key is not in the data dictionary, we skip it
				key = row[ID_idx]
				
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
				if truncated_code == None:
					continue

				if self.marshall_predictor(truncated_code, code_column):
					continue
				# if suffix == 'lab_results':
				# 	print(temp_min_val, temp_max_val)
				# 	val, min_val, max_val = self.make_lab_values(row[val_idx], temp_min_val, temp_max_val)
				# 	print(val, min_val, max_val)
				# 	if val == '':
				# 		continue
				
				# if in the required interval and code is valid
				if (begin <= date and date <= end) and pattern.match(truncated_code):
					# if we do not care about SOEPcode (always except for journaal case) or the SOEPcode is E
					if (not incorporate_SOEP) or (incorporate_SOEP and row[SOEP_idx] == 'E'):
					
						if 'lab_results' in suffix: # if we prepare for lab result abstraction							
							try:
								val = float(row[val_idx])
								min_val = minmax_dict[truncated_code]['low_bound']
								max_val = minmax_dict[truncated_code]['high_bound']

							except ValueError:
								continue

							if not 'ID2abstractions' in locals():
								# dict (patient) of dict (lab measurement name) of list of tuples (all value/date combinations of measurement)
								ID2abstractions = defaultdict(dict)
							
							util.init_key(ID2abstractions, key, defaultdict(dict))
							util.init_key(ID2abstractions[key], original_code, [])

							ID2abstractions[key][original_code].append((date, val))

							if '' not in [val, min_val, max_val]:
								attr = get_value(val, min_val, max_val, original_code)

								if not attr in attribute_count:
									attribute_count[attr] = 0

								# check if attribute name and ID instance already exist, if not, make them
								util.init_key(attribute2ids, attr, defaultdict(dict))
								util.init_key(attribute2ids[attr], key, 0)
								
								# add 1 to the occurrence of the attribute in the instance
								attribute2ids[attr][key] += 1
								attribute_count[attr] += 1

						else: # else no lab result collection, regular aggregation
							# generate attribute names

							if 'cardiometabolism' in suffix:
								val_idx = headers.index('valuec')
								value = str(row[val_idx])
							
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
								util.init_key(attribute2ids, attr, defaultdict(dict))
								util.init_key(attribute2ids[attr], key, 0)

								# add 1 to the occurrence of the attribute in the instance, except if attribute is binary
								if 'smoking' in suffix:
									if attribute2ids[attr][key] == 1:
										continue

								if 'allergies' in suffix:
									val_idx = headers.index('flag')
									value = row[val_idx]

									# check if the person actually has the allergie for which was tested
									if value == 'POS':
										attribute2ids[attr][key] = 1
									# if negative or not tested, it is assumed that person does not have particular allergie
									else:
										attribute2ids[attr][key] = 0

								else:
									attribute2ids[attr][key] += 1
									attribute_count[attr] += 1


							
				current += 1

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
							util.init_key(attribute2ids, attr, defaultdict(dict))
							util.init_key(attribute2ids[attr], ID, 0)
							attribute2ids[attr][ID] += 1
		# print len(attribute2ids)
		# print attribute2ids.keys()[0:5]
		
		

		# add data to each instance
		for ID in dct:
			data = dct[ID]['data']

			for id2occurrences in attribute2ids.values():
				
				# if patient has occurrences for the attribute, add that number, else add 0
				if ID in id2occurrences: 
					data.append(id2occurrences[ID])
				else:
					data.append(0)

		# return the keys to be used as headers when writing the processed data
		return list(attribute2ids.keys()), num_total, num_pos


	def marshall_predictor(self, code, src):
		# is_med_predictor = (src == 'atc_code') and code in ['A06','A07','B03']
		is_consult_predictor = (src == 'icpc_cat' or 'icpcprobleem') and code in [
			'K86', 'K87', 'T90' 'L88' 'K85', 'P17']
		is_lab_predictor = (src == 'dmemo') and code in [
			'CHOLBMT']
		is_smoke_predictor = (src == 'dmemo') and code in [
			'ROOKAQ', 'ADMIAQ', 'PAKJAQ']
		
		return is_smoke_predictor or is_consult_predictor or is_lab_predictor


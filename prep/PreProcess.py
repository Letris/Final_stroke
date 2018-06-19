import re
from prep.date_math import PatientInterval, str2date
from datetime import date,timedelta,datetime
import numpy as np
# from date_math import generate_patient_interval, generate_random_patient_interval, str2date
from util_.in_out import write_csv, save_obj, load_obj
import util_.util as util
import util_.sql as sql
from tqdm import *
import pandas as pd
from collections import defaultdict
from numba import jit
import logging
from prep.DataExtraction import DataExtraction

class PreProcess(DataExtraction):
	'''abstract class describing basic functionality of the preprocessing phase'''

	def __init__(self, in_dir, delim, out_dir, ID_column, min_age, max_age, interval, from_sql, HIS_list, survival, already_processed):
		self.in_dir = in_dir
		self.delim = delim # delimiter of input file (default ',')
		self.out_dir = out_dir
		self.ID_column = ID_column # name of the ID of a data instance
		self.min_age = min_age
		self.max_age = max_age 
		self.interval = interval # interval of data we deem relevant
		self.id2data = dict() # dict describing all data instances
		self.from_sql = from_sql
		# self.HIS_subquery = self.to_condition(HIS_list)
		self.survival = survival
		self.statistics = dict() # statistics of data 
		self.already_processed = already_processed

	def insert_data(self, f, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False):
		'''abstract method to be implemented by subclass'''
		print ('abstract method "insert_data" called by', type(self))

	def process(self, needs_processing):
		'''process using the specified source'''
		if self.from_sql:
			result = self.process_sql(needs_processing)
		else:
			result = self.process_csv(needs_processing)
			print(result)
		return result

	def process_csv(self, needs_processing):
		'''converts the specified csv's to usable data'''

		# get all csv's in the input folder
		self.files = util.list_dir_csv(self.in_dir)
		
		self.pickle_files = util.list_dir_pickle(self.in_dir)

		# put the IDs of the 'main' file in a dict		
		if self.already_processed == True:
			try:
				ID_f = util.select_file(self.pickle_files, 'patient_dict')
				self.id2data = load_obj(ID_f)
				self.headers = ['ID', 'age', 'gender']
				print('yyy')
			except TypeError:
				ID_f = util.select_file(self.files, 'patient')
				rows, fields = util.import_data(ID_f, delim=self.delim)		
				self.headers = self.get_IDs(rows, fields)

		else:
			ID_f = util.select_file(self.files, 'patient')
			rows, fields = util.import_data(ID_f, delim=self.delim)		
			self.headers = self.get_IDs(rows, fields)

			if self.survival == True:
				ID_f = util.select_file(self.files, 'icpc')
				rows, fields = util.import_data(ID_f, delim=self.delim)
				self.insert_start_baseline(rows, fields)

				
		# add stroke value to each patient
		if self.already_processed == True:
			try:
				stroke_f = util.select_file(self.pickle_files, 'stroke_dict')
				self.id2data = load_obj(stroke_f)
				print('xxx')
			
			except TypeError:
				stroke_f = util.select_file(self.files, 'icpc')
				rows, fields = util.import_data(stroke_f, delim=self.delim)
				self.get_stroke_occurrences(rows, fields)
			except ValueError:
				stroke_f = util.select_file(self.files, 'icpc')
				rows, fields = util.import_data(stroke_f, delim=self.delim)
				self.get_stroke_occurrences(rows, fields)

		else:
			# add stroke value to each patient
			stroke_f = util.select_file(self.files, 'icpc')
			rows, fields = util.import_data(stroke_f, delim=self.delim)
			self.get_stroke_occurrences(rows, fields)

		# randomize dates if non-survival
		if self.survival == False:
			self.insert_data_intervals()
		else:
			self.insert_survival_intervals()
			
		# gather data from medication csv
		if 'medication' in needs_processing and needs_processing['medication']:
			print('...processing medication')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('atc0_survival', 'atc0_headers0')
					else:
						self.load_data('atc_dict0', 'atc_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_medication()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_medication()

			else:
				self.process_medication()


					
		# gather data from consult csv
		if 'consults' in needs_processing and needs_processing['consults']:
			print('...processing consults')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('consults_dict0_survival', 'consults_headers0')
					else:
						self.load_data('consults_dict0', 'consults_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_consults()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_consults()

			else:
				self.process_consults()


		# gather data from verrichtingen csv
		if 'actions' in needs_processing and needs_processing['actions']:
			print('...processing actions')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('actions_dict0_survival', 'actions_headers0')
					else:
						self.load_data('actions_dict0', 'actions_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_actions()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_actions()

			else:
				self.process_actions()


		# gather data from icpc csv
		if 'icpc' in needs_processing and needs_processing['icpc']: #IS ALLEEN DEZE GESCHIKT VOOR TEMPORAL???
			print('...processing ICPC')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('icpc_dict0_survival', 'icpc_headers0')
					else:
						self.load_data('icpc_dict0', 'icpc_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_icpc()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_icpc()

			else:
				self.process_icpc()

 			
		# gather data from lab results csv
		if 'lab_results' in needs_processing and needs_processing['lab_results']:
			print('...processing lab results')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('lab_results_dict0_survival', 'lab_results_headers0')
					else:
						self.load_data('lab_results_dict0', 'lab_results_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_labresults()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_labresults()

			else:
				self.process_labresults()


		# gather data from smoking file
		if 'smoking' in needs_processing and needs_processing['smoking']:
			print('...processing smoking')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('smoking_dict0_survival', 'consults_headers0')
						self.load_data('smoking_dict1_survival', 'smoking_headers1')
					else:
						self.load_data('smoking_dict0', 'smoking_headers0')					
						self.load_data('smoking_dict1', 'smoking_headers1')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_smoking()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_smoking()

			else:
				self.process_smoking()


		if 'bmi' in needs_processing and needs_processing['bmi']:
			print('...processing bmi')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('bmi_dict0_survival', 'bmi_headers0')
						self.load_data('bmi_dict1_survival', 'bmi_headers1')
						self.load_data('bmi_dict2_survival', 'bmi_headers2')
					else:
						self.load_data('bmi_dict0', 'bmi_headers0')
						self.load_data('bmi_dict1', 'bmi_headers1')
						self.load_data('bmi_dict2', 'bmi_headers2')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_bmi()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_bmi()

			else:
				self.process_bmi()
		

		if 'allergies' in needs_processing and needs_processing['allergies']:
			print('...processing allergies')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('allergies_dict0_survival', 'allergies_headers0')
					else:
						self.load_data('allergies_dict0', 'allergies_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_allergies()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_allergies()

			else:
				self.process_allergies()

		
		if 'blood_pressure' in needs_processing and needs_processing['blood_pressure']:
			print('...processing blood pressure')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('blood_pressure_dict0_survival', 'blood_pressure_headers0')
						# self.load_data('blood_pressure_dict1_survival', 'blood_pressure_headers1')
					else:
						self.load_data('blood_pressure_dict0', 'blood_pressure_headers0')
						self.load_data('blood_pressure_dict1', 'blood_pressure_headers1')


				except TypeError:
						print('Data not available, processing medication data')
						self.process_bloodpressure()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_bloodpressure()

			else:
				self.process_bloodpressure()

			
		if 'alcohol' in needs_processing and needs_processing['alcohol']:
			print('...processing alcohol')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('alcohol_dict0_survival', 'alcohol_headers0')
					else:
						self.load_data('alcohol_dict0', 'alcohol_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_alcohol()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_alcohol()

			else:
				self.process_alcohol()


		if 'renal_function' in needs_processing and needs_processing['renal_function']:
			print('...processing renal function')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('renal_function_dict0_survival', 'renal_function_headers0')
					self.load_data('renal_function_dict0', 'renal_function_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_renalfunction()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_renalfunction()

			else:
				self.process_renalfunction()


		if 'cardiometabolism' in needs_processing and needs_processing['cardiometabolism']:
			print('...processing cardiometabolism')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('cardiometabolism_dict0_survival', 'renal_function_headers0')
					else:
						self.load_data('cardiometabolism_dict0', 'cardiometabolism_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_cardiometabolism()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_cardiometabolism()

			else:
				self.process_cardiometabolism()


		if 'lab_blood' in needs_processing and needs_processing['lab_blood']:
			print('...processing lab blood')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('lab_blood_dict0_survival', 'lab_blood_headers0')
					else:
						self.load_data('lab_blood_dict0', 'lab_blood_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_lab_blood()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_lab_blood()

			else:
				self.process_lab_blood()


		if 'lung_function' in needs_processing and needs_processing['lung_function']:
			print('...processing lung function')
			if self.already_processed == True:
				try:
					if self.survival == True:
						self.load_data('lung_function_dict0_survival', 'lung_function_headers0')
					else:
						self.load_data('lung_function_dict0', 'lung_function_headers0')

				except TypeError:
						print('Data not available, processing medication data')
						self.process_lung_function()
						
				except ValueError:
						print('Data not available, processing medication data')
						self.process_lung_function()

			else:
				self.process_lung_function()

		# move stroke indicator to end of each instance data list
		self.move_target_to_end_of_list()
		
		# append target element to headers, add to class var
		self.headers.append('target')
		# self.headers = headers

		to_remove = []

		for key, d in self.id2data.items():			
				date_info = d['stroke_dates']
				if self.survival == True:
					print(date_info[0])
					if not isinstance(date_info[0], list):
						if int(str(date_info[0]).split('-')[0]) < 2007:
							to_remove.append(key)
							continue

				else:
					if str(date_info[0]) != 'negative' :
						if int(str(date_info[0]).split('-')[0]) < 2007:
							to_remove.append(key)
							continue

		print(len(to_remove))
		for key in to_remove:
			del self.id2data[key] 	


	def get_IDs(self, rows, headers):
		'''sets all IDs as keys to a dict. Additionally adds gender/age data
			and date registration data'''
		print ('...getting all record IDs')

		# get the index of the relevant columns
		print(self.ID_column)
		print(headers)
		# ID_idx = headers.index(self.ID_column) #ID column index
		# age_idx = headers.index('birthyear') + 1 #age column index
		# gender_idx = headers.index('dgender') + 1 #gender column index
		# begin_idx = headers.index('dentrdate') + 1 #begin column index
		# end_idx = headers.index('dexitdate') + 1#end column index

		ID_amount = [] 
		too_young = []
		registration_none = []
		unregistration_none = []
		before_07 = 0
		avg_age = []
		
		max = 5000000000000000000000
		current = 0

		rows = rows.where((pd.notnull(rows)), None)

		# pair IDs with a dict corresponding to data and dates
		for row in tqdm(rows.itertuples()): #line in de data
			if current > max:
				break
			else:
				# key is ID
				if len(row) < 1:
					print('row < 1')
					break #zelf toegevoegd
				
				key = row.Index #int() weggehaald want key is ook met letters
				
				if key not in ID_amount:
					ID_amount.append(key)
				
				# skip if instance is outside the specified age limits
				try:
					if int(row.birthyear) > 2000:
						too_young.append(key)
						continue
				
				
					ID_age = 2018 - int(row.birthyear)
					avg_age.append(ID_age)

					# val is a new dict with keys 'data' en 'dates'
					# containing the processed data and registration dates, respectively
					val = dict()

					if self.survival == False:
						val['data'] = ['negative', key, ID_age, row.dgender] #key 'data'; values ['negative', ID, age, gender]
					else:
						val['data'] = [[False], key, ID_age, row.dgender]
					registration = str2date(row.dentrdate, give_default_begin=False) #registration date #default begin was true, even veranderd nav de pippi documenten				
					#str2date uit date_math.py; converts date to format dd/mm/yyyy

					unregistration = str2date(row.dexitdate, ymd=False, give_default_end=True) #if not (row[end_idx] in ['', None]) else str2date('2050-12-31')
					
					if registration == None:
						registration_none.append(key)
						continue
					if unregistration == None:
						unregistration_none.append(key)
						continue
					
					if int(str(unregistration).split('-')[0])<2007:
						before_07 += 1
						continue
					
					if self.survival == False:
						val['stroke_dates'] = ['negative', registration, unregistration] #key 'P_dates' ; values ['negative', begindate, enddate]
					else:
						val['stroke_dates'] = [[False], registration, unregistration]
					# add key/value pair
					self.id2data[key] = val #id2data dict; key=id, val=dict

				except ValueError:
					continue

				except TypeError:
					continue
				current +=1

		self.statistics['unique ids'] = len(ID_amount)
		self.statistics['too old ids'] = len(too_young)
		self.statistics['in database before study started'] = len(registration_none)
		self.statistics['in database before until'] = len(unregistration_none)
		self.statistics['in database before study started'] = before_07
		self.statistics['len id2data '] = len(self.id2data)
		self.statistics['average age'] = np.mean(avg_age)
		
		save_obj(self.id2data, self.in_dir + 'patient_dict')

		print('it worked!')
		return ['ID', 'age', 'gender']

	def get_stroke_occurrences(self, rows, headers):
		'''sets all stroke cases to initial diagnosis date values in 
			id2data[patient][stroke_dates][0]'''
		print ('...getting all target (stroke) occurrences')

		stroke_count = 0

		# get the index of the relevant columns
		stroke_idx = headers.index('icpc') + 1
		date_idx = headers.index('dicpc_startdate') + 1

		# regex patterns to match
		general_stroke_pattern = re.compile('K90')
		ischemic_stroke_pattern = re.compile('K90.03')
		intracerebral_hem_pattern = re.compile('K90.02')
		subarchnoid_hem_pattern = re.compile('K90.01')
		tia_stroke_pattern = re.compile('K89')

		max = 500000000000000000
		current = 0

		rows = rows.where((pd.notnull(rows)), None)

		# pair IDs with a dict corresponding to data and dates
		print(len(rows))

		for row in tqdm(rows.itertuples()):  #line in de data
			if current > max:
				break
			
			if row[date_idx] == " ": 
				continue

			else:
				# get key and if it's in the dict, the current corresponding stroke value

				key = row.Index
				if key in self.id2data:
					stroke = self.id2data[key]['stroke_dates'][0]
					# if self.survival == True and not isinstance(stroke, datetime.date):
					# 	stroke = stroke[0]

					# get ICPC code and its date
					code = row.icpc
					if code == None:
						continue
					elif type(code) == str:
						code = code.strip().upper()[0:3]

					code_date = str2date(date_str=row.dicpc_startdate, mdy=False, give_default_begin=True, give_default_end=True) #, mdy=False, give_default_begin=True, give_default_end=True

					
					# add stroke case if code matches, AND corresponding date is earlier than the currently recorded

					if self.survival:
						if (general_stroke_pattern.match(code) or ischemic_stroke_pattern.match(code) or intracerebral_hem_pattern.match(code) or subarchnoid_hem_pattern.match(code) or tia_stroke_pattern.match(code)):
								if (isinstance(stroke, list) and stroke[0] == False) or stroke > code_date:
									self.id2data[key]['stroke_dates'][0] = code_date
									self.id2data[key]['data'][0] = [True]
									stroke_count += 1
			
					if not self.survival:
						if (general_stroke_pattern.match(code) or ischemic_stroke_pattern.match(code) or intracerebral_hem_pattern.match(code)
							or subarchnoid_hem_pattern.match(code) or tia_stroke_pattern.match(code)) and (stroke == 'negative' or stroke > code_date):
							self.id2data[key]['stroke_dates'][0] = code_date		
							self.id2data[key]['data'][0] = 'positive'
							stroke_count += 1

				else:
					continue

			current+=1

		save_obj(self.id2data, self.in_dir + 'stroke_dict')
		self.statistics['stroke count'] = stroke_count
	
	def insert_start_baseline(self, rows, headers):

		dct = self.id2data
		rows = rows.where((pd.notnull(rows)), None)
		actions_dict = dict()

		code_idx = headers.index('icpc_cat') + 1
		date_idx = headers.index('dicpc_startdate') + 1

		# patterns = ['12000','12001', '12002', '12004']

		max = 5000000000000000000
		current = 0
		amount_x = 0
		amount_y = 0
		f=0
		g=0
		i=0
		z=0
		key_list=[]

		for row in tqdm(rows.itertuples()):
			current+=1

			if current > max:
				break

			key = row.Index
			
			if not key in dct:
				z+=1
				continue

			if not key in key_list:
				key_list.append(key)

			amount_x +=1

			date = str2date(row[date_idx], give_default_begin=True, give_default_end=True)

			if int(str(date).split('-')[0])<2007:
				continue

			original_code = row[code_idx]
			if original_code == None:
				i+=1
				continue
			
			string_code = str(original_code)

			if not key in actions_dict:
				actions_dict[key] = {}
				if not string_code in actions_dict[key]:
					actions_dict[key][string_code] = []
					actions_dict[key][string_code].append(date)
				else:
					actions_dict[key][string_code].append(date)

			else:
				if not string_code in actions_dict[key]:
					actions_dict[key][string_code] = []
					actions_dict[key][string_code].append(date)
				else:
					actions_dict[key][string_code].append(date)


		to_remove = []
		for patient, action_codes in actions_dict.items():
			amount_y+=1
			lowest_dict = dict()
			count = 0

			for action_code, dates in action_codes.items():
				if not dates:
					count+=1
					continue
				else:
					lowest_dict[action_code] = min(dates)
			
				# try:					
				# 	lowest_dict[action_code] = min(dates)
				# except ValueError:
				# 	continue
			# for action_code, date in lowest_dict:
			earliest_visit = min(lowest_dict, key=lowest_dict.get)
			visit_date = lowest_dict[earliest_visit]
			self.id2data[patient]['stroke_dates'].append(visit_date)
			print(self.id2data[patient]['stroke_dates'])
			# except ValueError:
			# 	to_remove.append(patient)

		print(amount_x, amount_y, z)
		print(f,g,i, g+f+i)
		print(len(to_remove))
		print(len(key_list))
		for key in to_remove:
			del self.id2data[key] 

	# def insert_start_baseline(self, rows, headers):

	# 	dct = self.id2data
	# 	rows = rows.where((pd.notnull(rows)), None)
	# 	actions_dict = dict()

	# 	code_idx = headers.index('prestatiecode') + 1
	# 	date_idx = headers.index('dverrdate') + 1

	# 	patterns = ['12000','12001', '12002', '12004']

	# 	max = 500000000000000
	# 	current = 0
	# 	amount_x = 0
	# 	amount_y = 0
	# 	f=0
	# 	g=0
	# 	i=0
	# 	z=0
	# 	key_list=[]

	# 	for row in tqdm(rows.itertuples()):
	# 		current+=1

	# 		if current > max:
	# 			break

	# 		key = row.Index
			
	# 		if not key in dct:
	# 			z+=1
	# 			continue

	# 		if not key in key_list:
	# 			key_list.append(key)

	# 		amount_x +=1

	# 		date = str2date(row[date_idx], give_default_begin=True, give_default_end=True)

	# 		original_code = row[code_idx]
	# 		if original_code == None:
	# 			i+=1
	# 			continue
			
	# 		string_code = str(int(original_code))

	# 		if not key in actions_dict:
	# 			actions_dict[key] = {'12000': [], '12001': [], '12002': [], '12004': []}
	# 			if string_code in patterns:
	# 				f+=1
	# 				actions_dict[key][string_code].append(date)
	# 			else:
	# 				g+=1
	# 		else:
	# 			if string_code in patterns:
	# 				f+=1
	# 				actions_dict[key][string_code].append(date)
	# 			else:
	# 				g+=1

		

	# 	to_remove = []
	# 	for patient, action_codes in actions_dict.items():
	# 		amount_y+=1
	# 		lowest_dict = dict()
	# 		count = 0

	# 		for action_code, dates in action_codes.items():
	# 			if not dates:
	# 				count+=1
	# 				continue
	# 			else:
	# 				lowest_dict[action_code] = min(dates)
				
	# 		if count == 4:
	# 			to_remove.append(patient)
	# 			continue
	# 			# try:					
	# 			# 	lowest_dict[action_code] = min(dates)
	# 			# except ValueError:
	# 			# 	continue
	# 		# for action_code, date in lowest_dict:
	# 		earliest_visit = min(lowest_dict, key=lowest_dict.get)
	# 		visit_date = lowest_dict[earliest_visit]
	# 		self.id2data[patient]['stroke_dates'].append(visit_date)
	# 		print(self.id2data[patient]['stroke_dates'])
	# 		# except ValueError:
	# 		# 	to_remove.append(patient)

	# 	print(amount_x, amount_y, z)
	# 	print(f,g,i, g+f+i)
	# 	print(len(to_remove))
	# 	print(len(key_list))
	# 	for key in to_remove:
	# 		del self.id2data[key] 

	def insert_data_intervals(self):
		'''per data instance, gets the intervals used within which the data is regarded'''
		print ('...getting all patient intervals to be used in the learning process')

		patient_interval = PatientInterval(10000)

		# iterate over dictionary
		to_remove = []
		for key, d in self.id2data.items():
			date_info = d['stroke_dates']

			# if the patient has no stroke, we randomize an interval. Else we pick exact dates
			if date_info[0] == 'negative':
				result = patient_interval.randomize(date_info[1], date_info[2], self.interval)
			else:
				result = patient_interval.calculate(date_info[1], date_info[0], self.interval)
			
			# if we were able to take an interval, append to date_info
			if result:
				date_info.append(result[0])
				date_info.append(result[1])
			else: # else set up for removal
				to_remove.append(key)

		# remove all keys in the dct for which an interval could not be generated
		for key in to_remove:
			del self.id2data[key] 


	# def generate_lab_attributes(self, original_code, suffix):
	# 	'''generates abstracted lab attributes, such as increasing HB, or low HB'''

			
	def insert_survival_intervals(self):
		print ('...getting all survival patient intervals to be used in the learning process')

		fixed_begin = date(2007,1,1)
		fixed_end = date(2017, 1, 1)
		to_remove = []
		x = 0
		k=0
		j=0
		o=0
		m=0
		for key, d in self.id2data.items():
				
				date_info = d['stroke_dates']
				print(date_info)
				data = d['data']
				if data[0][0] == True:
					x+=1
				stroke_date = date_info[0]
				begin_measurement = timedelta(days=self.interval[0])
				end_measurement = timedelta(days=self.interval[1])
				if isinstance(data[0], list):
					print('huh')
					# if patient did not have a stroke
					if data[0][0] == False: #data[0][0] == False:
						try:
							patient_begin = date_info[3]
							print('jay')
						except IndexError:
							print('nai')
							patient_begin = fixed_begin

						patient_exit_date = date_info[2]

						# if patient entered database before start of study, change start baseline to begin study
						if patient_begin < fixed_begin:
							patient_begin = fixed_begin
					
						# if patient did not exit database before end study, change to end of study
						if patient_exit_date == None:
							patient_exit_date = fixed_end

						# if the patient is in database shorter than baseline period, delete 
						if patient_begin + end_measurement >= patient_exit_date:
							to_remove.append(key)
						
						# define baseline period
						begin_baseline = patient_begin
						end_baseline = patient_begin + timedelta(days=self.interval[1])

						# add baseline period to data
						date_info.append(begin_baseline)
						date_info.append(end_baseline)

						# calculate follow-up time (survival in days)
						delta = patient_exit_date - end_baseline
						survival_time = delta.days
						data[0].append(survival_time)

					# if patient did have a stroke
					else:
						k +=1
						try:
							patient_begin = date_info[3]
						except IndexError:
							j+=1
							patient_begin = fixed_begin

						patient_exit_date = date_info[0]

						# if patient entered database before start of study, change start baseline to begin study
						if patient_begin < fixed_begin:
							patient_begin = fixed_begin

						if patient_exit_date < patient_begin:
							m+=1
							patient_begin = fixed_begin

						# if the patient is in database shorter than baseline period, delete 
						if patient_begin + end_measurement > patient_exit_date:
							print(patient_begin, end_measurement, patient_exit_date)
							o+=1
							to_remove.append(key)

						# define baseline period
						begin_baseline = patient_begin
						end_baseline = patient_begin + timedelta(days=self.interval[1])

						# add baseline period to data
						date_info.append(begin_baseline)
						date_info.append(end_baseline)

						# calculate follow-up time (survival in days)
						delta = patient_exit_date - end_baseline
						survival_time = delta.days
						data[0].append(survival_time)

		for key in to_remove:
			del self.id2data[key] 

		print('zz')
		print(len(to_remove))
		print(x)

		y=0

		for key, d in self.id2data.items():
				date_info = d['stroke_dates']
				data = d['data']

				if data[0][0] == True:
					y+=1
		
		print('ho')
		print(y)
		print('stop')
		print(k)
		print('jj')
		print(j)
		print(o)
		print(m)

		self.statistics['IDs removed, not enough time in db'] = len(to_remove)
		self.statistics['IDs left after removing from db'] = len(self.id2data)
	

	def append_known_data(self, data_to_append):
		for ID in data_to_append:
			ID_data = data_to_append[ID]
			if ID in self.id2data:
				for data in ID_data:
					self.id2data[ID]['data'].append(data)


	def load_data(self, data, headers):
		data_to_append = load_obj(util.select_file(self.pickle_files, data))
		print(data_to_append)
		self.append_known_data(data_to_append)
		new_headers = load_obj(util.select_file(self.pickle_files, headers))
		self.headers = self.headers + new_headers


	def generate_attributes(self, original_code, limit, suffix, value, src=''):
		'''Generate the attributes. In the most simple case
			this is a single attribute, namely the code + the 
			specified suffix.'''
		if value == None:
			return [self.generate_code(original_code, limit) + '_' + suffix[0]]
		else:
			return [self.generate_code(original_code, limit) + '_' + value.strip('0|') + '_' + suffix[0]]

	def move_target_to_end_of_list(self):
		'''moves first data value to end of list for each instance in data dictionary'''
		print ('...correctly positioning the target attribute')

		for k in self.id2data:
			data = self.id2data[k]['data']
			data.append(data.pop(0))

	def make_lab_values(self, val, min_val, max_val):
		try:
			val = float(val.replace(',', '.'))
		except ValueError:
			val = ''
		except AttributeError:
			val = ''
		try:
			min_val = float(min_val.replace(',', '.'))
		except ValueError:
			min_val = ''
		except AttributeError:
			min_val = ''
		try:
			max_val = float(max_val.replace(',', '.'))
		except ValueError:
			max_val = ''
		except AttributeError:
			max_val = ''
		return val, min_val, max_val

	def execute(self, cursor, query, query_footer=''):
		'''add HIS subsetting to query if it is available'''
		query = query + self.HIS_subquery + query_footer
		cursor.execute(query)

	# def to_condition(self, lst):
	# 	'''generate condition-segment of a query using the list lst'''
	# 	return " WHERE praktijkcode IN ('" + "','".join(lst) + "')"

	def	save_output(self, benchmark=False, sequence_file=False, sub_dir='', name='unnamed', target=False):
		'''saves processed data to the specified output directory'''
		print ('...saving processed data')# to {}'.format('sql' if self.from_sql else 'file')

		headers = self.headers
		# print (self.id2data.values())
		# print('x')
		# if we didn't get the data from sql database, just save to .csv
		if True or not self.from_sql:
			# possibly make new directories
			out_dir = self.out_dir + '/' + sub_dir + '/'
			util.make_dir(out_dir)

			f_out = out_dir + name + '.csv'
			out = write_csv(f_out)
			
			# write headers where required
			if benchmark:
				out.writerow(headers[0:3])
			elif target:
				out.writerow([headers[0], headers[-1]])
			elif sequence_file:
				pass
			else:
				out.writerow([headers[0]] + headers[3:-1])

			# write data
			for value in self.id2data.values():
				data = value['data']
				if benchmark:
					data = data[0:3]
					data[2] = 1 if data[2] == 'V' else 0
				elif target:
					if self.survival == False:
						data = [data[0], 0 if data[-1] == 'negative' else 1]
					else:
						data = [data[0], data[-1]]
				elif sequence_file:
					pass
				else:
					data = [data[0]] + data[3:-1]
				out.writerow(data)
		
	def save_statistics(self, sub_dir='data', name='unnamed'):
		out_dir = self.out_dir + '/' + sub_dir + '/'
		util.make_dir(out_dir)
		f_out = out_dir + name + '.csv'

		with open(f_out, 'w') as f:
    		 for key, value in self.statistics.items():
        		f.write('%s:%s\n' % (key, value))

	def find_minmax(self, values_dict, pattern, limit):

		minmax_dict = dict()
		for code, values in values_dict.items():
			
			# row = row.split(';')

			original_code = code
			if original_code == None:
				continue

			truncated_code = self.generate_code(original_code, limit)
			if truncated_code == None:
				continue


			if pattern.match(truncated_code):
				# if pattern not in important_predictors:
				value_list = np.array(values)
		# contextualize based on data distribution
		
				low_bound = np.percentile(value_list, 25)
				high_bound = np.percentile(value_list, 75)
				minmax_dict[truncated_code] = {'low_bound' : low_bound, 'high_bound' : high_bound}
		# print('huh')
		# max_val = max(value_list)
		# min_val = min(value_list)
		# print(max_val, min_val)

		return minmax_dict
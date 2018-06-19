from operator import attrgetter
import re

from prep.SequenceProcess import SequenceProcess
from prep.StateInterval import StateInterval
from prep.date_math import get_dates, str2date, four_weeks
import util_.util as util
import prep.abstracts
from collections import defaultdict

class NonMarshallSequenceProcess(SequenceProcess):
	'''class describing sequential/temporal way of preprocessing 
		the data by analyzing patterns of data concepts'''

	def insert_data(self, rows, headers, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False):
		'''inserts data from the specified csv and corresponding columns'''

		# make convenient reference to the dictionary
		dct = self.id2data

		# # get data and corresponding headers
		# rows, headers = util.import_data(f, delim=self.delim)

		# get the index of the relevant columns
		ID_idx = headers.index(self.ID_column)
		code_idx = headers.index(code_column)
		b_date_idx = headers.index(date_column[0])
		e_date_idx = headers.index(date_column[1])
		if suffix == 'lab_results':
			val_idx = headers.index('waarde')
			min_idx = headers.index('referentie_minimum')
			max_idx = headers.index('referentie_maximum')
		if incorporate_SOEP:
			SOEP_idx = headers.index(incorporate_SOEP)

		# get the right suffix to append for the attribute name
		if suffix == '':
			suffix = code_column

		# regex pattern to match (ATC/ICPC standards)
		pattern = re.compile(regex_string)

		max = 5000
		current = 0

		# iterate over all instances
		for row in rows:
			if current > max:
				break


			row = row.split(';')
			
			original_code = row[code_idx]
			if original_code == None:
				continue
			truncated_code = self.generate_code(original_code, limit) 
			if truncated_code == None:
				continue

			### is in Marshall Predictors check ###
			### if it is a marshall predictor, we skip this line.
			if self.marshall_predictor(truncated_code, code_column):
				continue

			# if key is not in the data dictionary, we skip it
			key = row[ID_idx]
			if not key in dct:
				continue

			# init other vars
			b_date = str2date(row[b_date_idx], give_default_begin=True) # begin of event
			e_date = str2date(row[e_date_idx], give_default_end=True) # end of event
			b_reg = dct[key]['stroke_dates'][1] # beginning of registration
			e_reg = dct[key]['stroke_dates'][2] # ending of registration
			if code_column == 'specialisme':
				e_reg = e_reg - four_weeks()

			if suffix == 'lab_results':
				val, min_val, max_val = self.make_lab_values(row[val_idx], row[min_idx], row[max_idx])
				if val == '':
					continue

			# if in the required interval (either beginning or ending date) AND code is valid
			if ( (b_reg <= b_date and b_date <= e_reg) or (b_reg <= e_date and e_date <= e_reg) ) and pattern.match(truncated_code):
				
				# if we need to take the SOEP code of consults into account
				if (not incorporate_SOEP) or (incorporate_SOEP and row[SOEP_idx] == 'E'):

					# generate attribute names
					if suffix == 'lab_results': # if we prepare for lab result abstraction
						if not 'ID2abstractions' in locals():
							# dict (patient) of dict (lab measurement name) of list of tuples (all value/date combinations of measurement)
							ID2abstractions = defaultdict(dict)
						
						util.init_key(ID2abstractions, key, defaultdict(dict))
						util.init_key(ID2abstractions[key], original_code, [])

						ID2abstractions[key][original_code].append((b_date, val))
					
						if '' not in [val, min_val, max_val]:
							attributes = [abstracts.get_value(val, min_val, max_val, original_code)]

							# # add value abstraction as state interval
							# self.insert_state_interval(key, attr, b_date, e_date)
						else:
							attributes = []

					else:
						attributes = self.generate_attributes(original_code, limit, suffix, src=code_column)

					# this loop allows multiple attributes to be created in the previous code line
					# this allows for other classes to subclass this class, e.g. SequenceEnrichProcess
					for attr in attributes:

						# insert a StateInterval object with the specified parameters
						self.insert_state_interval(key, attr, b_date, e_date, original_code, code_column)

			current += 1

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
						abstractions = abstracts.get_trends(k, points)
						for abstraction in abstractions:
							self.insert_state_interval(ID, *abstraction, original_code=original_code, src=code_column)
						# self.id2data[ID]['data'] = self.id2data[ID]['data'] + abstractions
		
		# to satisfy return value requirement for the method 'process' in the superclass
		return [], -1, -1
			
	def insert_state_interval(self, key, state, begin, end, original_code, src):
		'''converts state-begin-end-triples to state intervals, add to data record data'''
		sequence = self.id2data[key]['data']
		SI = StateInterval(state, begin, end)
		sequence.append(SI)

	def sort_sequences(self):
		'''sort each state sequence (= 1 per patient) consisting of state intervals
			order of sort is: begin date->end date->lexical order of state name'''
		for k in self.id2data:
			sequence = self.id2data[k]['data']
			static_seq = sequence[0:3] # gender/age
			dynamic_seq = sequence[3:-1]
			dynamic_seq.sort(key=attrgetter('begin', 'end', 'state'))
			stroke = [sequence[-1]]
			self.id2data[k]['data'] = static_seq + dynamic_seq + stroke
	
	def generate_code(self, code, limit):
		'''generates the required part of the code in a field, 
			e.g. atc code A01 in field A01B234'''
		result = code.upper().strip()[0:limit]
		if result.lower() in ['oncologie', 'chirurgie', 'gastro-enterologie', 'interne geneeskunde']:
			result = None
		#'interne geneeskunde           ','gastro-enterologie            ', 
		return result

	def marshall_predictor(self, code, src):
		is_med_predictor = (src == 'atc_code') and code in ['A06','A07','B03']
		is_consult_predictor = (src == 'icpc') and code in [
			'D01','D02','D06','D08','D11','D12','D15','D16','D18','D24','D93',
			'K93','K94',
			'T07','T08','T82','T89','T90','T92',
			'G04']
		is_lab_predictor = (src == 'codenorm') and code in [
			'HB','HBA1','HBA2',
			'MCH','MCHC','MCV'
			'OCCULTBLOE', 'OCBIDF']
		return is_med_predictor or is_consult_predictor or is_lab_predictor
		
if __name__ == '__main__':
	import sys

	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	age_range = (30,150)

	seq_p = SequenceProcess()
	seq_p.process(in_dir, out_dir, age_range)
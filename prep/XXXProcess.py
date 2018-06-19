import re

from prep.PreProcess import PreProcess
from prep.date_math import str2date
import util_.util as util
import prep.abstracts
from collections import defaultdict

class XXXProcess(StandardEnrichProcess):
	'''class describing standard way of preprocessing 
		the data by counting occurrences of data concepts'''

	def insert_data(self, rows, headers, code_column, date_column, regex_string, limit, suffix='', incorporate_SOEP=False):
		'''inserts data from the specified csv and corresponding columns'''

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
		
		# get the right suffix to append for the attribute name
		if suffix == '':
			suffix = code_column

		# regex pattern to match (ATC/ICPC standards)
		pattern = re.compile(regex_string)

		# iterate over all instances, making a new dict with the new attributes as keys
		attribute2counts = defaultdict(dict)
		for row in rows:

			# if key is not in the data dictionary, we skip it
			key = row[ID_idx]
			if not key in dct:
				continue

			# init other vars
			date = str2date(row[date_idx])
			begin = dct[key]['stroke_dates'][3]
			end = dct[key]['stroke_dates'][4]
			original_code = row[code_idx]
				
			# if we do not care about SOEPcode (always except for journaal case) or the SOEPcode is E
			if (not incorporate_SOEP) or (incorporate_SOEP and row[SOEP_idx] == 'E'):
			
				# generate attribute names
				attributes = self.generate_attributes(original_code, limit, suffix, src=code_column)
				
				# this loop allows multiple attributes to be created in the previous code line
				# this allows for other classes to subclass this class, e.g. StandardEnrichProcess
				for attr in attributes:

					# check if attribute name and ID instance already exist, if not, make them
					util.init_key(attribute2counts, attr, defaultdict(dict))
					util.init_key(attribute2counts[attr], key, 0)

					# add 1 to the occurrence of the attribute in the instance
					attribute2counts[attr] += 1
		
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
		return attribute2ids.keys()

	def generate_code(self, code, limit):
		'''generates the required part of the code in a field, 
			e.g. atc code A01 in field A01B234'''
		return code.upper().strip()[0:limit]

if __name__ == '__main__':
	dct = defaultdict(dict)
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
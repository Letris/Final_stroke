from prep.StandardProcess import StandardProcess
from prep.SequenceProcess import SequenceProcess
import util_.util as util
import util_.in_out as io

class StandardEnrichProcess(StandardProcess):
	'''Class describing standard way of preprocessing the data by 
		counting occurrences of enriched data concepts. Here, 
		enrichment indicates an abstraction of the original
		data is used, such as using side effects instead of
		medication.'''

	def __init__(self, in_dir, delim, out_dir, ID_column, min_age, max_age, interval, from_sql, HIS_list, mapping_files_dir):
		StandardProcess.__init__(self, in_dir, delim, out_dir, ID_column, min_age, max_age, interval, from_sql, HIS_list)
		
		# dict of enrichment dicts which describe the transformation from a code to some abstraction
		self.code2x_dict = fill_enrichment_dicts(mapping_files_dir)

	def generate_attributes(self, code, limit, suffix, src=''):
		'''Generate the attributes. In the case of 
			StandardEnrichProcess, this is a collection of 
			side effects, indications, and active ingredients.'''
		return generate_enriched_attributes(code, limit, suffix, src, self.code2x_dict, self.__class__.__bases__[0], self)

class SequenceEnrichProcess(SequenceProcess):
	'''Class describing sequential/temporal way of preprocessing 
		the data by analyzing patterns of enriched data concepts
		Here, enrichment indicates an abstraction of the original
		data is used, such as using side effects instead of
		medication.'''

	def __init__(self, in_dir, delim, out_dir, ID_column, min_age, max_age, interval, from_sql, HIS_list, mapping_files_dir):
		SequenceProcess.__init__(self, in_dir, delim, out_dir, ID_column, min_age, max_age, interval, from_sql, HIS_list)
		
		# list of enrichments dicts which describe the transformation from a code to some abstraction
		self.code2x_dict = fill_enrichment_dicts(mapping_files_dir)

	def generate_attributes(self, code, limit, suffix, src=''):
		'''Generate the attributes. In the case of 
			SequenceEnrichProcess, this is a collection of 
			side effects, indications, and active ingredients.'''
		return generate_enriched_attributes(code, limit, suffix, src, self.code2x_dict, self.__class__.__bases__[0], self)

	# def insert_state_interval(self, key, state, begin, end, original_code, src):
	# 	'''convert state-begin-end-triples to state intervals, 
	# 		making sure an incompatibility list
	# 		is added to the StateInterval to avoid explosion of search space 
	# 		when performing temporal mining'''
	# 	code2 = self.code2x_dict
	# 	code = original_code
		
	# 	incompatibility_list = [original_code]
	# 	if src == 'atc_code':
	# 		if code in code2['effects']:
	# 			incompatibility_list = incompatibility_list + code2['effects'][code]
	# 		if code in code2['indications']:
	# 			incompatibility_list = incompatibility_list + code2['indications'][code]
	# 		if code in code2['ingredients']:
	# 			incompatibility_list = incompatibility_list + code2['ingredients'][code]
	# 	elif src == 'icpc':
	# 		if code in code2['manifestationof']:
	# 			incompatibility_list = incompatibility_list + code2['manifestationof'][code]
	# 		if code in code2['associations']:
	# 			incompatibility_list = incompatibility_list + code2['associations'][code]

	# 	sequence = self.id2data[key]['data']
	# 	SI = StateInterval(state, begin, end, incompatibility_list)
	# 	sequence.append(SI)

def generate_enriched_attributes(code, limit, suffix, src, code2, ParentClass, process_instance):
	'''Generate enriched attributes.'''
	# which source file are we currently investigating?
	if src == 'atc_code': 
		# get all medication enrichments, adding a suffix to each attribute
		effects = util.get_dict_val(code2['effects'], code, default=[])
		effects = [effect + '_effect' for effect in effects]

		indications = util.get_dict_val(code2['indications'], code, default=[])
		indications = [indication + '_indication' for indication in indications]

		ingredients = util.get_dict_val(code2['ingredients'], code, default=[])
		ingredients = [ingredient + '_ingredient' for ingredient in ingredients]

		return effects + indications + ingredients + ParentClass.generate_attributes(process_instance, code, limit, suffix, src) # 'super' class call
	elif src == 'icpc':
		# print code, limit, suffix, src

		manifestationof = util.get_dict_val(code2['manifestationof'], code, default=[])
		manifestationof = [m + '_manifestationof' for m in manifestationof]

		association = util.get_dict_val(code2['association'], code, default=[])
		association = [a + '_association' for a in association]

		return manifestationof + association + ParentClass.generate_attributes(process_instance, code, limit, suffix, src) # 'super' class call
	# elif src == 'specialisme':
		# referral enrichments
	# elif src == 'omschrijving:'
		# icpc enrichments
	# elif src == 'codenorm':
		# lab result enrichments
	else:
		# default enrichment (= none, just return original value + suffix)
		#print 'no known associated enrichment possible with source {}. Returning original value and continuing'.format(src)
		return ParentClass.generate_attributes(process_instance, code, limit, suffix, src) # 'super' class call

def fill_enrichment_dicts(mapping_files_dir):
	'''loads the enrichment mappings'''

	# initiate result as a dict of dicts
	result = dict()

	# read frequent effects, put effects in dict, put dict in result
	try:
		rows = io.read_csv(mapping_files_dir + '/atc/effect_frequent.csv')
	except:
		rows = io.read_csv(mapping_files_dir + '/atc/effect.csv')		
	code2effects = {row[0] : row[1:] for row in rows}
	result['effects'] = code2effects

	# read frequent indications, put indications in dict, put dict in result
	try:
		rows = io.read_csv(mapping_files_dir + '/atc/indication_frequent.csv')
	except:
		try:
			rows = io.read_csv(mapping_files_dir + '/atc/indication.csv')
		except:
			rows = []
	code2indications = {row[0] : row[1:] for row in rows}
	result['indications'] = code2indications

	# read frequent ingredients, put ingredients in dict, put dict in result
	try:
		rows = io.read_csv(mapping_files_dir + '/atc/ingredient_frequent.csv')
	except:
		try:
			rows = io.read_csv(mapping_files_dir + '/atc/ingredient.csv')
		except:
			rows = []
	code2ingredients = {row[0] : row[1:] for row in rows}
	result['ingredients'] = code2ingredients

	# read frequent manifestations of symptoms, put manifestations in dict, put dict in result
	try:
		rows = io.read_csv(mapping_files_dir + '/icpc/manifestationof_frequent.csv')
	except:
		try:
			rows = io.read_csv(mapping_files_dir + '/icpc/manifestationof.csv')
		except:
			rows = []
	code2manifestation_of = {row[0] : row[1:] for row in rows}
	result['manifestationof'] = code2manifestation_of

	# read frequent associations of symptoms/disorders, put manifestations in dict, put dict in result
	try:
		rows = io.read_csv(mapping_files_dir + '/icpc/association_frequent.csv')
	except:
		try:
			rows = io.read_csv(mapping_files_dir + '/icpc/association.csv')
		except:
			rows = []
	code2association = {row[0] : row[1:] for row in rows}
	result['association'] = code2association

	return result


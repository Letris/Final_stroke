import util_.util as util
import util_.in_out as io
from prep.EnrichProcesses import StandardEnrichProcess
from math import sqrt
import scipy.stats as stats
from collections import defaultdict

class FrequencyCounter():

	def __init__(self, data_dir, export_dir, ALPHA):
		'''set in- and output directories, and which enrichment files 
			are simply skipped and not further pruned (only renamed)'''
		self.data_dir = data_dir
		self.export_dir = export_dir
		self.ALPHA = ALPHA

	def execute(self, subdirs, preprocess_args, needs_processing):
		'''process the enrichment files in the specified folder'''
		self.import_data(preprocess_args, needs_processing)
		self.count()
		self.significance_test()

		dirs = {subdir : (self.export_dir + '/' + subdir + '/') for subdir in subdirs}
		
		if 'atc' in dirs:
			self.export(dirs['atc'], 'effect')
			self.export(dirs['atc'], 'indication')
			# self.export(dirs['atc'], 'ingredient')
		if 'icpc' in dirs:
			self.export(dirs['icpc'], 'manifestationof')
			self.export(dirs['icpc'], 'association')

	def import_data(self, preprocess_args, needs_processing):
		sep = StandardEnrichProcess(*preprocess_args)
		sep.process(needs_processing)
		self.process_instance = sep

	def count(self):
		'''count each code among all files'''
		print ('...counting codes')
		headers = self.process_instance.headers
		id2data = self.process_instance.id2data
		self.abstraction2counts = defaultdict(dict)

		# print self.n1,self.n2
		for i, h in enumerate(headers):
			if h.lower() in ['id', 'target', 'age', 'gender']:
				continue
			count_neg = 0
			count_pos = 0 
			for ID in id2data:
				if int(id2data[ID]['data'][i]) > 0:
					if id2data[ID]['data'][-1] != 'negative':
						count_pos += 1
					else:
						count_neg += 1
			self.abstraction2counts[h] = [count_neg, count_pos]
			# print h, count_neg, count_pos, self.process_instance.num_med, self.process_instance.num_med_pos#, self.process_instance.num_cons, self.process_instance.num_cons_pos

		# for a in self.abstraction2counts:
		# 	if self.abstraction2counts[a] != (0, 0):
		# 		print self.abstraction2counts[a]
		# print len(self.abstraction2counts)
	
	def significance_test(self):
		'''test significance of each code among all files'''
		print ('...testing significance')
		id2data = self.process_instance.id2data
		n2 = float(sum((0 if lst['data'][-1] == 'negative' else 1 for lst in id2data.values()))) # num patients with positive stroke
		n1 = float(len(id2data) - n2) # num patients with negative stroke

		# remove keys whose ratio between the total and stroke-positive counts are over a threshold
		for key in self.abstraction2counts:

			_type = key.split('_')[-1]
			if not (_type in ['effect', 'indication', 'ingredient', 'manifestationof', 'association']):
				continue
			# if 'effect' in key or 'indication' in key or 'ingredient' in key:
			# 	n2 = float(self.process_instance.num_med_pos)
			# 	n1 = float(self.process_instance.num_med) - n2
			# elif 'manifestationof' in key or 'association' in key:
			# 	n2 = float(self.process_instance.num_cons_pos)				
			# 	n1 = float(self.process_instance.num_cons) - n2
			# else:
			# 	print 'error. unknown abstraction {} in FrequencyCounter.py, line 76ish'.format(key)
			# 	continue
			# 	# exit()

			# number of counts for total and positive categories
			neg_count, pos_count = self.abstraction2counts[key]

			# this snippet essentially corrects for double occurrences of abstractions in a single code (e.g. 2x 'Rash' in atc code A01B234)
			# if neg_count > n1: 
			# 	neg_count = n1
			# if pos_count > n2: 
			# 	pos_count = n2

			P_value = util.proporion_p_value(neg_count, n1, pos_count, n2)			
			self.abstraction2counts[key].append(P_value)

			# print '###'
			# print 'counts', neg_count, pos_count
			# print 'total', n1, n2
			# print 'proportions', p1, p2
			# print 'pss', p
			# print 'error', SE
			# print 'Z', Z
			# print 'P value', P_value
			# print

	def export(self, folder, suffix):
		'''export significant abstractions with the specified suffix to a new file'''
		print ('...exporting significance results')
		rows = io.read_csv(folder + suffix + '.csv')
		code2abstractions = {row[0] : row[1:] for row in rows}

		out = io.write_csv(folder + suffix + '_frequent.csv')
		for key, vals in code2abstractions.items(): 
			frequent_vals = []
			for abstraction in vals:
				suffixed_abstraction = abstraction+'_'+suffix
				if suffixed_abstraction in self.abstraction2counts and self.abstraction2counts[suffixed_abstraction][-1] < self.ALPHA:
					frequent_vals.append(abstraction)
				# if 'rectal discharge' in abstraction: 
				# 	print abstraction, suffixed_abstraction, suffixed_abstraction in self.abstraction2counts
				# 	print self.abstraction2counts.keys()[0]
				# 	print len(frequent_vals)
				# 	print self.abstraction2counts[suffixed_abstraction][-1], self.abstraction2counts[suffixed_abstraction][-1] < self.ALPHA

			if len(frequent_vals) > 0:
				out.writerow([key] + frequent_vals)

		# print self.abstraction2counts.keys()[0:50]
		# print self.abstraction2counts.values()[0:50]

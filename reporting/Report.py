import util_.in_out as io
import util_.util as util
from operator import itemgetter
from collections import defaultdict

class Report():
	'''provides the functionality to compile and export a report
		using various files resulting from previous steps in the process'''

	def __init__(self, f1, f2, f3, f_out, feature_threshold):
		'''initialize the report object'''
		self.f_general = f1
		self.f_data = f2
		self.f_predictors = f3
		self.f_out = f_out
		self.feature_threshold = feature_threshold

		self.compiled_result = defaultdict(dict)

	def compile(self):
		'''compile the individual parts of the report'''

		# compile general segment
		rows = io.read_csv2(self.f_general,delim=',')
		self.compiled_result['general'] = self.compile_general(rows)

		# compile header
		headers = ['predictor', '# stroke', '% stroke', '# No stroke', '% No stroke', '# Total', '% Total', 'P value', 'Model importance']
		self.compiled_result['headers'] = headers

		# compile results
		predictors = io.read_csv(self.f_predictors)
		data = io.read_csv(self.f_data)
		self.compiled_result['data'] = self.compile_data(predictors, data)

	def compile_general(self, rows):
		'''compile the general segment'''

		result = defaultdict(dict)

		# skip 8 rows
		for i in range(8):
			try:
				next(rows)
			except IOError as e:
				print (e, 'did you get the right file?')
				exit()

		# process info on 9th row
		stats = next(rows)
		self.num_stroke = int(float(stats[0]))
		self.num_Total = int(float(stats[1]))
		self.num_non_stroke = self.num_Total - self.num_stroke

		# save in dict
		result['headers'] = ['stroke', 'No stroke', 'Total']
		result['stats'] = [self.num_stroke, self.num_non_stroke, self.num_Total]
		return result

	def compile_data(self, predictors, data):
		'''compile data of relevant predictors'''

		# get relevant predictors
		relevant_predictors = [(p[0].lower(), float(p[1])) for p in predictors if abs(float(p[1])) >= self.feature_threshold]

		# get indices of the relevant predictors
		headers = next(data)
		relevant_tuples = [(i, h.lower()) for i, h in enumerate(headers) if h.lower() in zip(*relevant_predictors)[0]]

		# using the relevant indices, only keep the important data in memory
		relevant_data = defaultdict(dict)
		for d in data:
			# get 1 (stroke) or 0 (no stroke)
			target = int(d[-1])

			# get all relevant data of the instance
			relevant_instance_data = [int(d[i]) for i, h in relevant_tuples]

			# for every attribute, if the current instance has it at least once, add to the result dictionary
			if target in relevant_data:
				relevant_data[target] = [attr+1 if relevant_instance_data[i] > 0 else attr for i, attr in enumerate(relevant_data[target])]
			else:
				relevant_data[target] = [1 if relevant_instance_data[i] > 0 else 0 for i, attr in enumerate(relevant_instance_data)]

		# make it suitable for file output / human readable
		transposed_result = []
		for i, h in enumerate(relevant_tuples):

			# calc number of occurrences and percentage relative to population
			num_pos = relevant_data[1][i]
			per_pos = float(num_pos) / self.num_stroke *100
			num_neg = relevant_data[0][i]
			per_neg = float(num_neg) / self.num_non_stroke *100
			num_tot = num_pos + num_neg
			per_tot = float(num_tot) / self.num_Total *100

			# make list and append
			lst = [h[1], num_pos, per_pos, 
					num_neg, per_neg, 
					num_tot, per_tot, 
					util.proportion_p_value(num_neg,self.num_non_stroke,num_pos,self.num_stroke),
					relevant_predictors[i][1]]
			transposed_result.append(lst)

		# sort result by occurrence in the stroke column
		transposed_result.sort(key=itemgetter(1), reverse=True)
		return transposed_result

	def export(self):
		'''exports the result to the specified file'''
		# open file for writing
		out = io.write_csv(self.f_out)

		# write sources
		out.writerow(['general source', 'predictor source', 'data source'])
		out.writerow([self.f_general, self.f_predictors, self.f_data])
		out.writerow([])

		# write general stuff
		out.writerow(self.compiled_result['general']['headers'])
		out.writerow(self.compiled_result['general']['stats'])
		out.writerow([])

		# write headers to file
		out.writerow(self.compiled_result['headers'])

		# write individual results to file
		for row in self.compiled_result['data']:
			out.writerow(row)

if __name__ == '__main__':
	import sys
	args = sys.argv[1:]
	report = Report(args[0], args[1], args[2], args[3])
	report.compile()
	report.export()
import re

class EntityEnrichment():
	'''Abstract class for semantic enrichment'''

	def enrich(self, data_src):
		'''enrich input'''
		if data_src.lower() != 'sql':
			result = self.enrich_from_file(data_src)
		elif data_src.lower() == 'sql':
			result = self.enrich_from_sql()
		return result

	def filter_concepts(self, rows, idx, consider_SOEP=-1, regex_string=False, limit=None):
		'''returns a list of unique entity-values from the specified datafile'''
		
		# if we simply filter by unique values
		if consider_SOEP == -1 and not regex_string:
			return list(set([row[idx] for row in rows]))
		# if more sophisticated filtering is necessary
		else:
			soep_idx = consider_SOEP[0]
			soep_val = consider_SOEP[1]
			pattern = re.compile(regex_string)

			return list(set([row[idx][0:limit] for row in rows if row[soep_idx] == soep_val and pattern.match(row[idx])]))

	def merge_dicts(self, x2uri, uri2y):
		'''merge two dicts, where x2uri has uris as value, and uri2y as key'''
		to_remove = []
		for key in x2uri:
			if type(x2uri[key]) == list:
				uri_list = x2uri[key]
				try: 
					result_list = [uri2y[uri] for uri in uri_list]
					x2uri[key] = result_list
				except KeyError:
					# when no uri to x mapping was found
					# is a consequence of incomplete triple store
					# happens infrequently
					to_remove.append(key)
			else:
				try:
					x2uri[key] = uri2y[x2uri[key]]
				except KeyError:
					to_remove.append(key)

		for key in to_remove:
			del x2uri[key]

		print ('removed {} out of {} keys when merging dicts'.format(len(to_remove), len(x2uri)+len(to_remove)))

		result = x2uri
		return result
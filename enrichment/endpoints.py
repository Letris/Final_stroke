from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
from urllib.error import HTTPError, URLError
import time

class AbstractEndpoint(SPARQLWrapper):
	'''abstract object adding functionality to the extended SPARQLWrapper'''
	
	MAX_QUERY_SIZE = 1000 # maximum allowable query size of (most) open SPARQL endpoints
	EXECUTE_WAIT = 10 # time in seconds to wait before retrying query after encountering a http error

	def __init__(self, url, query_head, query_foot):
		'''init SPARQLWrapper with the specified URL, set the return format,
			set the size max_body_size of the parameters of our batch query'''
		SPARQLWrapper.__init__(self, url)
		self.setReturnFormat(JSON)
		self.query_head = query_head
		self.query_foot = query_foot
		self.max_body_size = self.MAX_QUERY_SIZE - len(self.query_head) - len(self.query_foot)
		
	def list2str(self, l, prepend='', append='', start_at=0, final_ignore=0):
		'''converts a list of str to a single string with str prepend and append before
			after each element, with a maximum length of max_body_size '''
		
		if '{}' in append: 
			replace = True # then we replace any {} with the current element
		else:
			replace = False

		result = ''
		for i, el in enumerate(l[start_at:]):
			append_replaced = append if not replace else append.replace('{}', el)
			if len(result + prepend + str(el) + append_replaced) > self.max_body_size: # 10% margin to eb sure
				break
			result = result + prepend + str(el) + append_replaced

		if final_ignore > 0: result = result[:-final_ignore] # remove segment
		return result, i + start_at

	def execute(self):
		'''execute the query set by SPARQLWrapper.setQuery(q)'''
		try:
			results = self.query().convert()
		except HTTPError as e:
			# print traceback.format_exc()
			print ('server overload. trying again in {} seconds'.format(self.EXECUTE_WAIT))
			print ('error msg:', e)
			time.sleep(self.EXECUTE_WAIT)
			results = self.execute()
		except ValueError as e:
			print ('server had trouble handling the request, but we\'re trying again!')
			print ('error msg:', e)
			time.sleep(self.EXECUTE_WAIT)
			results = self.execute()
		except URLError as e:
			print (e, '...trying again')
			time.sleep(self.EXECUTE_WAIT)
			results = self.execute()
		except Exception as e:
			# print 'unknown error occurred. waiting and continuing'
			print ('error msg:', e, '... waiting and continuing')
			time.sleep(self.EXECUTE_WAIT)
			results = self.execute()
		print ('retrieved results and returning')
		return results

	def add(self, dct, key, val):
		'''adds a val to a key in a dict depending on the type'''
		if type(dct[key]) == list:
			dct[key].append(val)
		elif type(dct[key]) == unicode:
			dct[key] = val
		else:
			print ('err: encountered type {} for dict value {} in endpoints.py'.format(type(val), val))
			exit()

	def set_prepend(self, s):
		self.prepend = s

	def set_append(self, s):
		self.append = s

class BioPortal(AbstractEndpoint):
	'''BioPortal subclass of the extended AbstractEndpoint'''
	
	def __init__(self, query_head, query_foot):
		'''instantiate a BioPortal object '''
		url = 'http://sparql.bioontology.org/sparql/'
		AbstractEndpoint.__init__(self, url, query_head, query_foot)
		
		api_key = "YOUR API KEY" # shouldn't be necessary, else make an account on the bioportal website
		self.addCustomParameter("apikey",api_key)
		
	def batch_query(self, concepts, dict_type):
		'''queries BioPortal for uri to side effects'''
		print ('...querying BioPortal')

		uri2x = defaultdict(dict_type)	
		i = 0
		while i < len(concepts):
			concepts_str, i = self.list2str(concepts, prepend='icpc:', append=',', start_at=i, final_ignore=1)

			q = self.query_head + concepts_str + self.query_foot
			self.setQuery(q)
			results = self.execute()

			for result in results['results']['bindings']:
				key = result['key']['value'].split('/')[-1]
				val = result['val']['value']
				if dict_type == unicode:
					if not 'dailymed' in val: 
						continue
					val = val.split('/')[-1]

				self.add(uri2x, key, val)
			
			i += 1

		return uri2x

class Dailymed(AbstractEndpoint):
	'''Dailymed subclass of the extended AbstractEndpoint'''

	def __init__(self, query_head, query_foot):
		'''instantiate a Dailymed object '''
		url = 'http://wifo5-04.informatik.uni-mannheim.de/dailymed/sparql'
		AbstractEndpoint.__init__(self, url, query_head, query_foot)

	def batch_query(self, uris, dict_type):
		'''queries Dailymed'''
		print ('...querying Dailymed')

		result_dict = dict()					
		i = 0
		while i < len(uris):
			uris_str, i = self.list2str(uris, prepend=self.prepend, append=self.append, start_at=i, final_ignore=1)

			q = self.query_head + uris_str + self.query_foot
			self.setQuery(q)
			results = self.execute()

			for result in results['results']['bindings']:
				key = result['key']['value'].split('/')[-1]
				val = result['val']['value']
				result_dict[key] = val
			# print float(i)/len(uris)
			
			i += 1
		return result_dict

class DBPedia(AbstractEndpoint):
	'''DBPedia subclass of the extended AbstractEndpoint'''
	
	def __init__(self):
		url = 'http://dbpedia.org/sparql'
		AbstractEndpoint.__init__(self, url)

class Diseasome(AbstractEndpoint):
	'''Diseasome subclass of the extended AbstractEndpoint'''

	def __init__(self, query_head, query_foot):
		'''instantiate a diseasome object '''
		url = 'http://wifo5-04.informatik.uni-mannheim.de/diseasome/sparql'
		AbstractEndpoint.__init__(self, url, query_head, query_foot)

	def batch_query(self, uris):
		'''queries diseasome for uri to disease'''
		print ('...querying diseasome')

		uri2disease = dict()					
		i = 0
		while i < len(uris):
			uris_str, i = self.list2str(uris, prepend='disease:', append=',', start_at=i, final_ignore=1)

			q = self.query_head + uris_str + self.query_foot
			self.setQuery(q)
			results = self.execute()

			for result in results['results']['bindings']:
				key = result['uri']['value'].split('/')[-1]
				val = result['disease']['value']
				uri2disease[key] = val
			
			i += 1
		return uri2disease

class Drugbank(AbstractEndpoint):
	'''Drugbank subclass of the extended AbstractEndpoint'''

	def __init__(self, query_head, query_foot):
		'''instantiate a drugbank object '''
		url = 'http://wifo5-04.informatik.uni-mannheim.de/drugbank/sparql'
		AbstractEndpoint.__init__(self, url, query_head, query_foot)

	def batch_query(self, concepts, dict_type):
		'''queries drugbank for atc to disease uris'''
		print ('...querying drugbank')

		code2disease_uri = defaultdict(dict_type)
		disease_uris = set()
		i = 0
		while i < len(concepts):
			concepts_str, i = self.list2str(concepts, prepend='"', append='",', start_at=i,  final_ignore=1)

			q = self.query_head + concepts_str + self.query_foot
			self.setQuery(q)
			results = self.execute()
			
			code2disease_uri, disease_uris = self.process(results, code2disease_uri, disease_uris)

			i += 1

		return code2disease_uri, list(disease_uris)

	def process(self, json, dct, uris):
		'''process the JSON dict'''
		for result in json['results']['bindings']:
			key = result['atc']['value']
			val = result['uri']['value'].split('/')[-1]
			self.add(dct, key, val)
			uris.add(val)
		return dct, uris

class Sider(AbstractEndpoint):
	'''Sider subclass of the extended AbstractEndpoint'''
	
	def __init__(self, query_head, query_foot):
		'''instantiate a sider object '''
		url = 'http://wifo5-04.informatik.uni-mannheim.de/sider/sparql'
		AbstractEndpoint.__init__(self, url, query_head, query_foot)
		self.MAX_QUERY_SIZE = 500

	def batch_query(self, uris, dict_type):
		'''queries sider for uri to side effects'''
		print ('...querying sider for {} concepts'.format(len(uris)))

		uri2x = defaultdict(dict_type)	
		new_uris = set()		
		i = 0
		while i < len(uris):
			uris_str, i = self.list2str(uris, prepend=self.prepend, append=self.append, start_at=i, final_ignore=6)

			q = self.query_head + uris_str + self.query_foot
			self.setQuery(q)
			results = self.execute()
			print ('... query {} percent complete'.format(float(i)/len(uris)*100))

			for result in results['results']['bindings']:
				key = result['key']['value'].split('/')[-1]
				val = result['val']['value']
				if dict_type == unicode:
					if not 'dailymed' in val: 
						continue
					val = val.split('/')[-1]

				self.add(uri2x, key, val)
				new_uris.add(val)
			
			i += 1

		return uri2x, list(new_uris)

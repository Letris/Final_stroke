import util_.in_out as io
import util_.util as util
import enrichment.endpoints
from enrichment.EntityEnrichment import EntityEnrichment

class ATC(EntityEnrichment):

	def export(self, out_dir):
		'''export results'''
		util.make_dir(out_dir)
		
		io.dict2csv(self.code2indications, out_dir + 'indication.csv')
		io.dict2csv(self.code2effects, out_dir + 'effect.csv')
		io.dict2csv(self.code2ingredients, out_dir + 'ingredient.csv')

	def enrich_from_file(self, in_dir):
		'''enrich using a data file as source'''
		assert(in_dir != '')
		files = util.list_dir_csv(in_dir)
		med_f = util.select_file(files, 'medicatie')
		records = io.read_csv(med_f)

		headers = util.get_headers(next(records))
		idx = headers.index('atc_code')

		return self.atc_enrichment(records, idx)

	def enrich_from_sql(self):
		'''enrich using a database as source'''
		cursor = util.sql_connect().cursor()
		cursor.execute('''SELECT atc_code FROM medicaties''')
		idx = 0 # we query only ATC codes
		return self.atc_enrichment(cursor, idx)

	def atc_enrichment(self, rows, idx):
		concepts = self.filter_concepts(rows, idx)

		self.code2indications = self.indication_enrichment(concepts)
		self.code2effects = self.side_effects_enrichment(concepts)
		self.code2ingredients = self.active_ingredient_enrichment(concepts)

		return self.code2indications, self.code2effects, self.code2ingredients

	def indication_enrichment(self, concepts):
		'''queries drugbank and diseasome for the atc --> indication step'''
		print ('### indication enrichment')
		
		drugbank_query_head = '''PREFIX drugbank: <http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugbank/>
					SELECT ?atc ?uri WHERE { ?drug drugbank:atcCode ?atc .
					?drug drugbank:possibleDiseaseTarget ?uri .
					FILTER (?atc IN ('''
		drugbank_query_foot = '))}'
		drugbank = endpoints.Drugbank(drugbank_query_head, drugbank_query_foot)
		code2uri, uris = drugbank.batch_query(concepts, dict_type=list)

		diseasome_query_head = '''PREFIX disease: <http://wifo5-04.informatik.uni-mannheim.de/diseasome/resource/diseases/>
								 PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
								 SELECT * WHERE {?uri rdfs:label ?disease . FILTER (?uri IN ('''
		diseasome_query_foot = '))}'
		diseasome = endpoints.Diseasome(diseasome_query_head, diseasome_query_foot)
		uri2indication = diseasome.batch_query(uris)

		code2disease = self.merge_dicts(code2uri, uri2indication)

		return code2disease

	def side_effects_enrichment(self, concepts):
		'''queries drugbank and sider for the atc --> side effects step'''
		print ('### side effects enrichment')

		drugbank_query_head = '''PREFIX drugbank: <http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugbank/>
					SELECT ?atc ?uri WHERE { ?uri drugbank:atcCode ?atc .
					FILTER (?atc IN ('''
		drugbank_query_foot = '))}'
		drugbank = endpoints.Drugbank(drugbank_query_head, drugbank_query_foot)
		code2uri, uris = drugbank.batch_query(concepts, dict_type=unicode)

		sider_query_head = '''PREFIX drugbank: <http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugs/>
					PREFIX owl: <http://www.w3.org/2002/07/owl#>
					PREFIX sider: <http://wifo5-04.informatik.uni-mannheim.de/sider/resource/sider/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					SELECT ?key ?val WHERE {'''
		sider_query_foot = '}'
		sider = endpoints.Sider(sider_query_head, sider_query_foot)
		sider.set_prepend('{?drug owl:sameAs drugbank:')
		sider.set_append(' . ?drug sider:sideEffect ?effect . ?effect rdfs:label ?val . BIND (drugbank:{} AS ?key)} UNION')
		uri2effects, _ = sider.batch_query(uris, dict_type=list)

		code2effects = self.merge_dicts(code2uri, uri2effects)

		return code2effects

	def active_ingredient_enrichment(self, concepts):
		'''queries drugbank, sider, and dailymed for the atc --> active ingredients step'''	
		print ('### ingredient enrichment')

		drugbank_query_head = '''PREFIX drugbank: <http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugbank/>
					SELECT ?atc ?uri WHERE { ?uri drugbank:atcCode ?atc .
					FILTER (?atc IN ('''
		drugbank_query_foot = '))}'
		drugbank = endpoints.Drugbank(drugbank_query_head, drugbank_query_foot)
		code2uri, uris = drugbank.batch_query(concepts, dict_type=unicode)

		# print u'J05AB01', code2uri[u'J05AB01']

		sider_query_head = '''PREFIX owl: <http://www.w3.org/2002/07/owl#>
					PREFIX drug: <http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugs/>
					SELECT DISTINCT ?key ?val WHERE {?siderdrug owl:sameAs ?key. ?siderdrug owl:sameAs ?val FILTER (?key != ?val)
					FILTER (?key IN('''
		sider_query_foot = '))}'
		sider = endpoints.Sider(sider_query_head, sider_query_foot)
		sider.set_prepend('drug:')
		sider.set_append(',')
		
		uri2uri, uris_sider = sider.batch_query(uris, dict_type=unicode)

		# print u'DB00787', uri2uri[u'DB00787']

		dailymed_query_head = '''PREFIX dailymed: <http://wifo5-04.informatik.uni-mannheim.de/dailymed/resource/dailymed/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					PREFIX drug: <http://wifo5-04.informatik.uni-mannheim.de/dailymed/resource/drugs/>
					select ?key ?val where {
					?key dailymed:activeIngredient ?ingredient. ?ingredient rdfs:label ?val.
					FILTER (?key IN ('''
		dailymed_query_foot = '))}'
		dailymed = endpoints.Dailymed(dailymed_query_head, dailymed_query_foot)
		dailymed.set_prepend('drug:')
		dailymed.set_append(',')
		uri2ingredient = dailymed.batch_query(uris_sider, dict_type=unicode)

		# print u'76', uri2ingredient[u'76']

		code2uri_sider = self.merge_dicts(code2uri, uri2uri)
		
		# print u'J05AB01', code2uri_sider[u'J05AB01']

		code2ingredients = self.merge_dicts(code2uri_sider, uri2ingredient)

		# print u'J05AB01', code2uri_sider[u'J05AB01']

		return code2ingredients

if __name__ == '__main__':
	# in terminal: python integrate.py ../Desktop/indication.csv ../Desktop/effects.csv ../Desktop/ingredients.csv

	# ... or import from SQL (uncomment)
	import cx_Oracle
	rows = cx_Oracle.connect('datamart', 'datamart', '10.67.201.10:1521/XE').cursor()
	rows.execute('''SELECT med.atc_code
						FROM AA_KOP_patient_dates dts 
						LEFT JOIN medicaties med ON dts.patientnummer = med.patientnummer''')
	idx = 0 # we only query atc codes

	# enrich
	indications, effects, ingredients = atc_enrichment(rows, idx)
	
	# export
	io.dict2csv(indications, sys.argv[1])
	io.dict2csv(effects, sys.argv[2])
	io.dict2csv(ingredients, sys.argv[3])


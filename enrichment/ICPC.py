import util_.in_out as io
import util_.util as util
import enrichment.endpoints
from enrichment.EntityEnrichment import EntityEnrichment
import re

class ICPC(EntityEnrichment):

	def export(self, out_dir):
		'''export results'''
		util.make_dir(out_dir)
		
		io.dict2csv(self.code2manifestation_of, out_dir + 'manifestationof.csv')
		io.dict2csv(self.code2association, out_dir + 'association.csv')

	def enrich_from_file(self, in_dir):
		'''enrich using a data file as source'''
		assert(in_dir != '')
		files = util.list_dir_csv(in_dir)
		med_f = util.select_file(files, 'journaal')
		records = io.read_csv(med_f)

		headers = util.get_headers(next(records))
		idx = headers.index('icpc')

		return self.icpc_enrichment(records, idx)

	def enrich_from_sql(self):
		'''enrich using a database as source'''
		cursor = util.sql_connect().cursor()
		cursor.execute('''SELECT substr(icpc,1,3) FROM journalen WHERE soepcode = 'E' ''')
		idx = 0 # we query only ICPC codes
		return self.icpc_enrichment(cursor, idx, from_sql=True)

	def icpc_enrichment(self, rows, idx, from_sql=False):
		if from_sql:
			pattern = re.compile('[A-Z][0-9][0-9]')
			concepts = self.filter_concepts(rows, idx)
			concepts = [c for c in concepts if pattern.match(c)]
		else:
			concepts = self.filter_concepts(
				rows, idx, consider_SOEP=(0,'E'), regex_string='[A-Z][0-9][0-9]', limit=3)

		self.code2manifestation_of = self.manifestation_of_enrichment(concepts)
		self.code2association = self.association_enrichment(concepts)

		return self.code2manifestation_of, self.code2association

	def manifestation_of_enrichment(self, concepts):
		'''queries bioportal for the icpc --> manifestation_of step'''
		print ('### manifestation enrichment')
		
		bp_query_head = '''PREFIX snomed: <http://purl.bioontology.org/ontology/SNOMEDCT/>
PREFIX icpc: <http://purl.bioontology.org/ontology/ICPC/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX umls: <http://bioportal.bioontology.org/ontologies/umls/>
SELECT DISTINCT ?key ?val
FROM <http://bioportal.bioontology.org/ontologies/ICPC>
FROM <http://bioportal.bioontology.org/ontologies/SNOMEDCT>
WHERE
{
    ?key umls:cui ?cui.
    ?s umls:cui ?cui.
    ?s snomed:definitional_manifestation_of ?o.
    ?o skos:prefLabel ?val.
    FILTER (?key IN ('''
		bp_query_foot = '))}'
		bp = endpoints.BioPortal(bp_query_head, bp_query_foot)
		code2manifestation_of = bp.batch_query(concepts, dict_type=list)

		return code2manifestation_of

	def association_enrichment(self, concepts):
		'''queries bioportal for the icpc --> association step'''
		print ('### association enrichment')
		
		lower_concepts = [c for c in concepts if int(c[1:]) < 30]
		upper_concepts = list(set(concepts) - set(lower_concepts))

		query_head = '''PREFIX snomed: <http://purl.bioontology.org/ontology/SNOMEDCT/>
PREFIX icpc: <http://purl.bioontology.org/ontology/ICPC/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX umls: <http://bioportal.bioontology.org/ontologies/umls/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?key ?val
FROM <http://bioportal.bioontology.org/ontologies/ICPC>
FROM <http://bioportal.bioontology.org/ontologies/SNOMEDCT>
WHERE
{
	?key umls:cui ?cui.
	?concept umls:cui ?cui.'''

		lower_bound_query_head = query_head + '''
	?child rdfs:subClassOf ?concept.
	?child skos:altLabel ?val.
	FILTER CONTAINS(?val, "(disorder)")
	FILTER (?key IN ('''

		upper_bound_query_head = query_head + '''
	?concept rdfs:subClassOf ?parent.
	?parent skos:altLabel ?val.
	FILTER CONTAINS(?val, "(finding)")
	FILTER (?key IN ('''

		bp_query_foot = '))}'

		bp = endpoints.BioPortal(lower_bound_query_head, bp_query_foot)
		code2association_lower = bp.batch_query(lower_concepts, dict_type=list)
		
		bp = endpoints.BioPortal(upper_bound_query_head, bp_query_foot)
		code2association_upper = bp.batch_query(upper_concepts, dict_type=list)

		code2association_lower.update(code2association_upper)
		code2association = code2association_lower
		return code2association


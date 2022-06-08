from SPARQLWrapper import SPARQLWrapper, JSON
import json


class BioFeatureAnnotation(object):

    REQUIRED_FIELD = 'name'
    PREFIX = {}
    QUERIES = {}

    def __init__(self, sparql_endpoint):
        self.sparql_endpoint = sparql_endpoint

    def __run_query__(self, query):
        pass

    def __call__(self, sparql_fields, nodes, *args, **kwargs):
        pass


class SampleAnnotation(object):

    REQUIRED_FIELD = 'sampleName'

    PREFIX = {'geo': '<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>'}

    QUERIES = {
        'experiment': '''
            PREFIX geo: {geo_prefix}
            SELECT ?s ?o WHERE {{
                ?s <http://purl.obolibrary.org/obo/OBI_0001896> ?o .
                FILTER (
                    {subjects}
                )
            }}''',
        'cultivar': '''
            PREFIX geo: {geo_prefix}
            SELECT ?s ?o WHERE {{
                ?s <http://www.ebi.ac.uk/efo/EFO_0005136> ?o .
                FILTER (
                    {subjects}
                )
            }}
        ''',
        'rootstock': '''
            PREFIX geo: {geo_prefix}
            SELECT ?s ?o WHERE {{
                ?s <http://purl.obolibrary.org/obo/NCBITaxon_580088> ?o .
                FILTER (
                    {subjects}
                )
            }}
        ''',
        'general_qualifier': '''
            PREFIX geo: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>
            PREFIX obo: <http://purl.obolibrary.org/obo/>
            
            SELECT ?s ?o WHERE {{
              ?s <http://purl.obolibrary.org/obo/NCIT_C27993> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
        ''',
        'genotype': '''
            PREFIX geo: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>
            PREFIX obo: <http://purl.obolibrary.org/obo/>
            
            SELECT ?s ?o WHERE {{
              ?s <http://purl.obolibrary.org/obo/NCIT_C16631> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
        ''',
        'plant_maturity': '''
            PREFIX geo: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>
            PREFIX obo: <http://purl.obolibrary.org/obo/>
            
            SELECT ?s ?o WHERE {{
              ?s <http://purl.obolibrary.org/obo/FOODON_03530050> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
        ''',
        'dev_stage': '''
            PREFIX geo: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>
            PREFIX obo: <http://purl.obolibrary.org/obo/>
            
            SELECT ?s ?o WHERE {{
              ?s <http://purl.obolibrary.org/obo/PO_0007033> ?bn1 .
              ?bn1 <https://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?o .
              FILTER (
                {subjects}
              )
            }}
        ''',
        'tissue': '''
            PREFIX geo: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=>
            PREFIX obo: <http://purl.obolibrary.org/obo/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT distinct ?s ?o WHERE {{
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                SERVICE <http://sparql.hegroup.org/sparql/> {{
                    ?x rdfs:label ?o
                }}
                FILTER (
                    {subjects}
                )
                FILTER(STRSTARTS(STR(?x), "http://purl.obolibrary.org/obo/PO_"))
            }}
        '''
    }

    def __init__(self, sparql_endpoint):
        self.sparql_endpoint = sparql_endpoint

    def __run_query__(self, query):
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return [triple for triple in results['results']['bindings']]

    def __call__(self, sparql_fields, nodes, *args, **kwargs):
        nodes_mapping = {
            self.PREFIX['geo'][1:-1] + n[self.REQUIRED_FIELD].replace('.ch1', '').replace('.ch2', ''): n for n in nodes
        }
        for field in sparql_fields:
            if field not in self.QUERIES.keys():
                continue
            for n in nodes:
                n[field] = None
            subjects = ' || '.join('?s=<' + n + '>' for n in nodes_mapping.keys())
            query = self.QUERIES[field].format(subjects=subjects, geo_prefix=self.PREFIX['geo'])
            triples = self.__run_query__(query)
            for triple in triples:
                _id = triple['s']['value']
                _value = triple['o']['value']
                for k, v in self.PREFIX.items():
                    _value = _value.replace(v[1:-1], '')
                nodes_mapping[_id][field] = _value


class SampleSetAnnotation(object):

    REQUIRED_FIELD = 'design'

    QUERIES = {
        "tissue": None
    }

    def __init__(self, sparql_endpoint):
        self.sparql_endpoint = sparql_endpoint

    def __run_query__(self, query):
        pass

    def __call__(self, sparql_fields, nodes, *args, **kwargs):
        for node in nodes:
            design = json.loads(node['design'])
            if design['elements']['edges']: # contrast
                pass
            else: # only condition
                # get all samples
                # check if the field is the same in all samples
                pass
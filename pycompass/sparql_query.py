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

    PREFIX = {
        'geo': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=',
        'sra': 'https://www.ncbi.nlm.nih.gov/sra?term=',
        'obo': 'http://purl.obolibrary.org/obo/',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
    }

    QUERIES_FILTER = {
        'tissue': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT distinct ?s ?o WHERE {{ {{
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                SERVICE <http://sparql.hegroup.org/sparql/> {{
                    ?x rdfs:label ?o
                    FILTER (
                       {objects}
                    )
                }}
                FILTER(STRSTARTS(STR(?x), "http://purl.obolibrary.org/obo/PO_"))
            }} UNION {{
                ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
                ?s1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                SERVICE <http://sparql.hegroup.org/sparql/> {{
                    ?x rdfs:label ?o
                    FILTER (
                       {objects}
                    )
                }}
                FILTER(STRSTARTS(STR(?x), "http://purl.obolibrary.org/obo/PO_"))
            }}
        }}
        '''
    }

    QUERIES = {
        'experiment': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
                ?s <http://purl.obolibrary.org/obo/OBI_0001896> ?o .
                FILTER (
                    {subjects}
                )
            }} UNION {{
                ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
                ?s1 <http://purl.obolibrary.org/obo/OBI_0001896> ?o .
                FILTER (
                    {subjects}
                )
            }} }}''',
        'cultivar': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
                ?s <http://www.ebi.ac.uk/efo/EFO_0005136> ?o .
                FILTER (
                    {subjects}
                ) 
            }} UNION {{
                ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
                ?s1 <http://www.ebi.ac.uk/efo/EFO_0005136> ?o .
                FILTER (
                    {subjects}
                )
            }} }}
        ''',
        'rootstock': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
                ?s <http://purl.obolibrary.org/obo/NCBITaxon_580088> ?o .
                FILTER (
                    {subjects}
                )
            }} UNION {{
                ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
                ?s1 <http://purl.obolibrary.org/obo/NCBITaxon_580088> ?o .
                FILTER (
                    {subjects}
                )
            }}
            }}
        ''',
        'general_qualifier': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
              ?s <http://purl.obolibrary.org/obo/NCIT_C27993> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }} UNION {{
              ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
              ?s1 <http://purl.obolibrary.org/obo/NCIT_C27993> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
            }}
        ''',
        'genotype': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
              ?s <http://purl.obolibrary.org/obo/NCIT_C16631> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }} UNION {{
              ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
              ?s1 <http://purl.obolibrary.org/obo/NCIT_C16631> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
            }}
        ''',
        'plant_maturity': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
              ?s <http://purl.obolibrary.org/obo/FOODON_03530050> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }} UNION {{
              ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
              ?s1 <http://purl.obolibrary.org/obo/FOODON_03530050> ?x .
              SERVICE <http://sparql.hegroup.org/sparql/> {{
                ?x rdfs:label ?o
              }}
              FILTER (
                {subjects}
              )
            }}
            }}
        ''',
        'dev_stage': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT ?s ?o WHERE {{ {{
              ?s <http://purl.obolibrary.org/obo/PO_0007033> ?bn1 .
              ?bn1 <https://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?o .
              FILTER (
                {subjects}
              )
            }} UNION {{
              ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
              ?s1 <http://purl.obolibrary.org/obo/PO_0007033> ?bn1 .
              ?bn1 <https://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?o .
              FILTER (
                {subjects}
              )
            }}
            }}
        ''',
        'tissue': '''
            PREFIX geo: {geo_prefix}
            PREFIX obo: {obo_prefix}
            PREFIX sra: {sra_prefix}
            PREFIX rdfs: {rdfs_prefix}
            
            SELECT distinct ?s ?o WHERE {{ {{
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                SERVICE <http://sparql.hegroup.org/sparql/> {{
                    ?x rdfs:label ?o
                }}
                FILTER (
                    {subjects}
                )
                FILTER(STRSTARTS(STR(?x), "http://purl.obolibrary.org/obo/PO_"))
            }} UNION {{
                ?s1 <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?s .
                ?s1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                SERVICE <http://sparql.hegroup.org/sparql/> {{
                    ?x rdfs:label ?o
                }}
                FILTER (
                    {subjects}
                )
                FILTER(STRSTARTS(STR(?x), "http://purl.obolibrary.org/obo/PO_"))
            }}
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
            self.PREFIX['geo'] + n[self.REQUIRED_FIELD].replace('.ch1', '').replace('.ch2', ''): n for n in nodes
        }
        nodes_mapping.update({
            self.PREFIX['sra'] + n[self.REQUIRED_FIELD]: n for n in nodes
        })
        for field in sparql_fields:
            if field not in self.QUERIES.keys():
                continue
            for n in nodes:
                n[field] = None
            subjects = ' || '.join('?s=<' + n + '>' for n in nodes_mapping.keys())
            query = self.QUERIES[field].format(
                subjects=subjects,
                obo_prefix='<' + self.PREFIX['obo'] + '>',
                geo_prefix='<' + self.PREFIX['geo'] + '>',
                sra_prefix='<' + self.PREFIX['sra'] + '>',
                rdfs_prefix='<' + self.PREFIX['rdfs'] + '>'
            )
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
        "tissue": None,
        "plant_maturity": None,
        "dev_stage": None,
        "genotype": None,
        "experiment": None,
        "cultivar": None,
        "rootstock": None,
        "general_qualifier": None
    }

    def __init__(self, sparql_endpoint):
        self.sparql_endpoint = sparql_endpoint

    def filter(self, key, values, *args, **kwargs):
        graphql_query = kwargs['graphql_query']
        query = SampleAnnotation.QUERIES_FILTER[key]
        objects = ' || '.join('?o="' + v + '"' for v in values)
        query = query.format(
            objects=objects,
            obo_prefix='<' + SampleAnnotation.PREFIX['obo'] + '>',
            geo_prefix='<' + SampleAnnotation.PREFIX['geo'] + '>',
            sra_prefix='<' + SampleAnnotation.PREFIX['sra'] + '>',
            rdfs_prefix='<' + SampleAnnotation.PREFIX['rdfs'] + '>'
        )
        triples = self.__run_query__(query)
        sample_acc_ids = set()
        for triple in triples:
            sample_acc_ids.add(triple['s']['value'].replace(SampleAnnotation.PREFIX['geo'], '').replace(SampleAnnotation.PREFIX['sra'], ''))
        samples = graphql_query._compendium.query('samples').filter(sampleName_In=list(sample_acc_ids))
        sample_sets = graphql_query._compendium.query('sampleSets').filter(samples=[str(s.id) for s in samples])
        return [ss.id for ss in sample_sets]

    def __run_query__(self, query):
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return [triple for triple in results['results']['bindings']]

    def __limma_normalization__(self, design, sparql_fields, compendium):
        ref = design['elements']['edges'][0]['data']['source']
        ref_samples = [x['data']['name'] for x in
                       filter(lambda x: x['data'].get('parent', None) == ref, design['elements']['nodes'])]
        ref_sample_nodes = [{SampleAnnotation.REQUIRED_FIELD: sample} for sample in ref_samples]

        test = design['elements']['edges'][0]['data']['target']
        test_samples = [x['data']['name'] for x in
                       filter(lambda x: x['data'].get('parent', None) == test, design['elements']['nodes'])]
        test_sample_nodes = [{SampleAnnotation.REQUIRED_FIELD: sample} for sample in test_samples]

        annotation = SampleAnnotation(self.sparql_endpoint)
        annotation(sparql_fields, ref_sample_nodes)
        annotation(sparql_fields, test_sample_nodes)

        ref_annotation = {field: sample.get(field, None) for sample in ref_sample_nodes for field in sparql_fields}
        test_annotation = {field: sample.get(field, None) for sample in test_sample_nodes for field in sparql_fields}
        contrast_annotation = {}

        for field in sparql_fields:
            contrast_annotation[field + '_test'] = test_annotation[field]
            contrast_annotation[field + '_ref'] = ref_annotation[field]
        return contrast_annotation

    def __tpm_normalization__(self, design, sparql_fields, compendium):
        samples = set([x['data']['name'] for x in filter(lambda x: x['data']['type'] == 'sample', design['elements']['nodes'])])
        sample_nodes = [{SampleAnnotation.REQUIRED_FIELD: sample} for sample in samples]

        annotation = SampleAnnotation(self.sparql_endpoint)
        annotation(sparql_fields, sample_nodes)

        return sample_nodes[0]

    def __call__(self, sparql_fields, nodes, *args, **kwargs):
        compendium = kwargs.get('compendium', None)

        for node in nodes:
            design = json.loads(node['design'])
            if compendium.normalization == 'limma':
                annotation = self.__limma_normalization__(design, sparql_fields, compendium)
                del node['design']
                node.update(annotation)
            elif compendium.normalization == 'tpm':
                del node['design']
                annotation = self.__tpm_normalization__(design, sparql_fields, compendium)
                node.update(annotation)

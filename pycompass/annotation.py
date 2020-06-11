from pycompass.biological_feature import BiologicalFeature
from pycompass.query import query_getter, run_query
from pycompass.sample_set import SampleSet
from pycompass.sample import Sample


class Annotation:
    '''
    The Annotation class wraps a BiologicalFeature or a Sample object to return its annoation
    '''

    def __init__(self, obj):
        self.compendium = None
        self.obj = obj
        if type(obj) != list:
            self.obj = [obj]
        for i in self.obj:
            if i.compendium:
                self.compendium = i.compendium
                break

    def get_triples(self):
        '''
        Return the annotation as a list of RDF triples

        :return: list
        '''
        ids = []
        for o in self.obj:
            if type(o) == BiologicalFeature or type(o) == Sample:
                ids.append(o.id)
        if len(ids) == 0:
            raise Exception("There's no valid Sample objects")
        query = '''{{
          annotationPrettyPrint(compendium:"{compendium}", ids:[{ids}]) {{
            rdfTriples
          }}
        }}'''.format(compendium=self.compendium.compendium_name, ids=','.join(['"' + id + '"' for id in ids]))
        json = run_query(self.compendium.connection.url, query)
        return json['data']['annotationPrettyPrint']['rdfTriples']

    def plot_network(self, output_format='html'):
        '''
        Return the Cytoscape JS representation of RDF graph used to annotate the BiologialFeature or Sample passed to the Annotation object

        :param output_format: html or json
        :return:
        '''
        ids = []
        for o in self.obj:
            if type(o) == BiologicalFeature or type(o) == Sample:
                ids.append(o.id)
        if len(ids) == 0:
            raise Exception("There's no valid BiologicalFeature or Sample objects to plot")
        query = '''{{
          annotationPrettyPrint(compendium:"{compendium}", ids:[{ids}]) {{
            {output}
          }}
        }}'''.format(compendium=self.compendium.compendium_name, ids=','.join(['"' + id + '"' for id in ids]),
                     output=output_format)
        json = run_query(self.compendium.connection.url, query)
        return json['data']['annotationPrettyPrint'][output_format]

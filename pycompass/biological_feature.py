from pycompass.utils import get_compendium_object
from pycompass.query import query_getter, run_query


class BiologicalFeature:
    '''
    A BiologicalFeature object represent the measured biological entity (tipically gene expression)
    '''

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if k == 'biofeaturevaluesSet':
                for n in v['edges']:
                    field = n['node']['bioFeatureField']['name']
                    value = n['node']['value']
                    setattr(self, field, value)
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        if name not in self.__dict__:
            _bf = BiologicalFeature(**dict({'compendium': self.compendium}, **BiologicalFeature.__get_biological_features(self.compendium, filter={'id': self.id})[0]))
            self.__dict__.update(_bf.__dict__)
        value = self.__dict__.get(name)
        if not value:
            raise AttributeError({self.__class__.__name__} + '.' + name + ' is invalid.')
        return value

    def by(self, *args, **kwargs):
        '''
        Get BiolgicalFeature list from other high level objects

        :param args:
        :param kwargs: sparql="SELECT ?s ?p ?o ..."
        :return:
        '''
        if 'sparql' in kwargs:
            sparql = kwargs['sparql']
            query = '''{{
                        sparql(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                            query:"{query}", target:"biofeature") {{
                            rdfTriples
                      }}
                    }}'''.format(compendium=self.compendium.compendium_name,
                                 version=self.compendium.version,
                                 database=self.compendium.database,
                                 normalization=self.compendium.normalization,
                                 query=sparql)
            json = run_query(self.compendium.connection.url, query)
            ids = set()
            for triple in json['data']['sparql']['rdfTriples']:
                ids.update(triple)
            ids.remove(None)
            filter = {'id_In': list(ids)}
            return self.get(filter=filter)
        elif 'annotationTerm' in kwargs:
            term = kwargs['annotationTerm']
            query = '''{{
              biofeatureAnnotations(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                annotationTerm:"{term}") {{
                edges {{
                  node {{
                    bioFeature {{
                      id
                    }}
                  }}
                }}
              }}
            }}'''.format(compendium=self.compendium.compendium_name,
                         version=self.compendium.version,
                         database=self.compendium.database,
                         normalization=self.compendium.normalization,
                         term=term)
            json = run_query(self.compendium.connection.url, query)
            ids = set()
            for n in json['data']['biofeatureAnnotations']['edges']:
                ids.add(n['node']['bioFeature']['id'])
            filter = {'id_In': list(ids)}
            return self.get(filter=filter)
        elif 'ontologyId' in kwargs:
            term = kwargs['ontologyId']
            query = '''{{
                          biofeatureAnnotations(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                            ontologyId:"{term}") {{
                            edges {{
                              node {{
                                bioFeature {{
                                  id
                                }}
                              }}
                            }}
                          }}
                        }}'''.format(compendium=self.compendium.compendium_name,
                                     version=self.compendium.version,
                                     database=self.compendium.database,
                                     normalization=self.compendium.normalization,
                                     term=term)
            json = run_query(self.compendium.connection.url, query)
            ids = set()
            for n in json['data']['biofeatureAnnotations']['edges']:
                ids.add(n['node']['bioFeature']['id'])
            filter = {'id_In': list(ids)}
            return self.get(filter=filter)

    @query_getter('biofeatures',
                  ['id', 'name', 'description',
                   'biofeaturevaluesSet { edges { node { bioFeatureField { name }, value } } }'])
    @staticmethod
    def __get_biological_features(obj, filter=None, fields=None):
        pass

    def get(self, filter=None, fields=None):
        '''
        Get biological feature

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of BiologicalFeature objects
        '''
        bf_ids = BiologicalFeature.__get_biological_features(self.compendium, filter=filter, fields=['id', 'name', 'description'])
        return [BiologicalFeature(**dict({'compendium': self.compendium}, **bf)) for bf in bf_ids]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @staticmethod
    def using(compendium):
        cls = get_compendium_object(BiologicalFeature, aggregate_class='biofeatures')
        return cls(compendium=compendium)

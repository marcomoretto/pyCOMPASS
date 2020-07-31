from pycompass.experiment import Experiment
from pycompass.platform import Platform
from pycompass.query import query_getter, run_query
from pycompass.utils import get_compendium_object


class Sample:
    '''
    The Sample class
    '''

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if k == 'experiment':
                self.__experiment_id__ = v['id']
            elif k == 'platform':
                self.__platform_id__ = v['id']
            else:
                setattr(self, k, v)

    @property
    def experiment(self):
        return Experiment.using(self.compendium).get(filter={'id': self.__experiment_id__})[0]

    @property
    def platform(self):
        return Platform.using(self.compendium).get(filter={'id': self.__platform_id__})[0]

    def by(self, *args, **kwargs):
        '''
        Get samples by using another object
        Example: Sample.using(compendium).by(platform=plt)

        :return: list of Sample objects
        '''
        if 'experiment' in kwargs:
            filter = {'experiment_ExperimentAccessId': kwargs['experiment'].experimentAccessId}
            return self.get(filter=filter)
        elif 'platform' in kwargs:
            filter = {'platform_PlatformAccessId': kwargs['platform'].platformAccessId}
            return self.get(filter=filter)
        elif 'annotationTerm' in kwargs:
            term = kwargs['annotationTerm']
            query = '''{{
              sampleAnnotations(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                annotationTerm:"{term}") {{
                edges {{
                  node {{
                    sample {{
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
            for n in json['data']['sampleAnnotations']['edges']:
                ids.add(n['node']['sample']['id'])
            filter = {'id_In': list(ids)}
            return self.get(filter=filter)
        elif 'ontologyId' in kwargs:
            term = kwargs['ontologyId']
            query = '''{{
                          sampleAnnotations(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                            ontologyId:"{term}") {{
                            edges {{
                              node {{
                                sample {{
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
            for n in json['data']['sampleAnnotations']['edges']:
                ids.add(n['node']['sample']['id'])
            filter = {'id_In': list(ids)}
            return self.get(filter=filter)
        elif 'sparql' in kwargs:
            sparql = kwargs['sparql']
            query = '''{{
                sparql(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                    query:"{query}", target:"sample") {{
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
        else:
            return []

    def get(self, filter=None, fields=None):
        '''
        Get compendium samples

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of Sample objects
        '''
        @query_getter('samples', ['id', 'sampleName', 'description', 'platform { id, platformAccessId }',
                                         'experiment { id, experimentAccessId }'])
        def _get_samples(obj, filter=None, fields=None):
            pass
        return [Sample(**dict({'compendium': self.compendium}, **e)) for e in _get_samples(self.compendium, filter=filter, fields=fields)]

    @staticmethod
    def using(compendium):
        cls = get_compendium_object(Sample, aggregate_class='samples')
        return cls(compendium=compendium)

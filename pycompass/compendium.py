####
# compendium.query('experiments').fields
# compendium.query('experiments').related_objects
# compendium.query('experiments').filter(experimentAccessId='GSExxx').fields(['description'])
# compendium.query('experiments').from(samples=samples).fields(['description'])
# compendium.module(genes=['VIT_xxx'], contrasts=['name'])
# compendium.query('sample').from(sparql="SELECT *")
# compendium.query('sample').from(annotation={'tissue'='leaf')) #compendium.query('sample').from(annotation={'tissue'='PO_xxx', cultivar:'Cabernet Sauvignon'})
# module.retr
#
import pycompass
from pycompass.graphql_query import QuerySet
from pycompass.module import Module
import pandas as pd
import numpy as np

DEFAULT_GRAPHQL_ENDPOINT = "http://fempc0734:2424/graphql" #"http://compass.fmach.it/graphql"
DEFAULT_BASE_URL = "http://fempc0734:2424/" #"http://compass.fmach.it/static"
DEFAULT_SPARQL_ENDPOINT = "http://compass.fmach.it/sparql"
DEFAULT_COMPENDIUM = "vespucci"

class Compendium(object):
    ALLOWED_OBJECTS = {
        'experiments': 'ExperimentType',
        'dataSources': 'DataSourceType',
        'biofeatures': 'BioFeatureType',
        'sampleSets': 'SampleSetType',
        'scoreRankMethods': 'RankMethodType',
        'ranking': 'RankingType'
    }

    def __init__(self, url=DEFAULT_GRAPHQL_ENDPOINT, name=DEFAULT_COMPENDIUM, version=None, database=None, normalization=None):
        self.url = url
        self.name = name
        self.version = version
        self.database = database
        self.normalization = normalization
        self.__modules_n__ = 0

        is_valid = False
        for compendium in self.available_compendium:
            is_valid = is_valid or name == compendium.name
            if not version:
                version = compendium.defaultVersion
            try:
                version = list(filter(lambda x: x['versionNumber'] == version or x['versionAlias'] == version, compendium.versions))[0]
                self.version = version['versionAlias']
            except Exception as e:
                raise Exception('Invalid version')
            if not database:
                database = version['defaultDatabase']
            try:
                database = list(filter(lambda x: x['name'] == database, version['databases']))[0]
                self.database = database['name']
            except Exception as e:
                raise Exception('Invalid database')
            all_normalizations = []
            for norm in database['normalizations']:
                _norm = norm.replace('(default)', '').strip()
                all_normalizations.append(_norm)
                is_default = '(default)' in norm
                if not normalization and is_default:
                    normalization = _norm
                self.normalization = normalization
            if normalization not in all_normalizations:
                raise Exception('Invalid normalization')

        if not is_valid:
            raise Exception('Invalid compendium name')

    @property
    def whole_compendium(self):
        connections = []
        if self.version:
            connections.append('version:"{}"'.format(self.version))
        if self.database:
            connections.append('database:"{}"'.format(self.database))
        if self.normalization:
            connections.append('normalization:"{}"'.format(self.normalization))
        query = '''
        {{
            wholeCompendium(compendium:"{compendium}") {{
                staticFilename
            }}
        }}
        '''.format(
            compendium=self.name,
            connections=', ' + ','.join(connections)
        )
        qs = QuerySet(self, None)
        qs._object_type = 'compendia'
        json = qs.__run_query__(query)
        file_url = DEFAULT_BASE_URL + json['data']['wholeCompendium']['staticFilename']
        df = pd.read_csv(file_url, compression='zip')
        df = df.set_index('Unnamed: 0')
        df.index.names = ['biofeatures']
        df.columns.names = ['samplesets']
        return Module(compendium=self, df=df, name='Whole compendium')

    @property
    def available_compendium(self):
        query = '''
        {
          compendia{
            name,
            fullName,
            description,
            defaultVersion,
            versions {
              versionNumber,
              versionAlias,
              defaultDatabase,
              databases {
                name,
                normalizations
              }
            }
          }
        }
        '''
        qs = QuerySet(self, None)
        qs._object_type = 'compendia'
        json = qs.__run_query__(query)
        return qs.__create_namedtuples__(json['data']['compendia'])

    def query(self, object_type):
        for k, v in self.ALLOWED_OBJECTS.items():
            if object_type and object_type.lower() == k.lower():
                return QuerySet(self, k)
        raise Exception('You cannot query this object. Use the GraphQL interface.')

    def module(self, *args, **kwargs):
        ss = kwargs.get('samplesets', [])
        bf = kwargs.get('biofeatures', [])
        qs = QuerySet(self, 'modules')

        try:
            bf = [x.id for x in bf]
            ss = [x.id for x in ss]
        except AttributeError as e:
            pass

        if bf:
            qs.filter(
                biofeaturesIds=','.join(bf)
            )
        if ss:
            qs.filter(
                samplesetIds=','.join(ss)
            )
        query = qs.__create_query_module__()
        json = qs.__run_query__(query)

        bf_idx = [x['node']['id'] for x in json['data']['modules']['biofeatures']['edges']]
        ss_idx = [x['node']['id'] for x in json['data']['modules']['sampleSets']['edges']]
        values = np.array(json['data']['modules']['normalizedValues'])

        df = pd.DataFrame(values, index=bf_idx, columns=ss_idx)
        df.index.name = 'biofeatures'
        df.columns.name = 'samplesets'

        # sort module
        connections = []
        if self.version:
            connections.append('version:"{}"'.format(self.version))
        if self.database:
            connections.append('database:"{}"'.format(self.database))
        if self.normalization:
            connections.append('normalization:"{}"'.format(self.normalization))
        query = '''
        {{
          plotHeatmap(compendium:"{compendium}", plotType:"module_heatmap_expression", biofeaturesIds:[{biofeatures}], samplesetIds:[{samplesets}] {connections}) {{
                sortedSamplesets {{
                  id
                }},
                sortedBiofeatures {{
                  id
                }},
                submoduleLineBiofeatures,
                submoduleLineSamplesets
            }}
        }}       
        '''.format(
            compendium=self.name,
            connections=', ' + ','.join(connections),
            biofeatures=','.join('"{}"'.format(x) for x in bf_idx),
            samplesets=','.join('"{}"'.format(x) for x in ss_idx)
        )
        json = qs.__run_query__(query)
        bf_idx = [x['id'] for x in json['data']['plotHeatmap']['sortedBiofeatures']]
        ss_idx = [x['id'] for x in json['data']['plotHeatmap']['sortedSamplesets']]

        cluster_biofeatures = [x for x in json['data']['plotHeatmap']['submoduleLineBiofeatures']]
        cluster_samplesets = [x for x in json['data']['plotHeatmap']['submoduleLineSamplesets']]

        df = df.loc[bf_idx][ss_idx]

        bf = kwargs.get('biofeatures', None)
        try:
            fields = bf._fields.copy()
            fields.remove('id')
            fields = list(fields)
            df[fields] = pd.DataFrame([[getattr(x, field) for field in fields] for x in bf], index=[x.id for x in bf])
            df = df.reset_index().set_index(fields + ['biofeatures'])
        except Exception as e:
            pass

        ss = kwargs.get('samplesets', None)
        try:
            fields = ss._fields.copy()
            fields.remove('id')
            fields = list(fields)
            df = df.T
            df[fields] = pd.DataFrame([[getattr(x, field) for field in fields] for x in ss], index=[x.id for x in ss])
            df = df.reset_index().set_index(fields + ['samplesets'])
            df = df.T
        except Exception as e:
            pass

        # create multiindex clusters
        bf_clusters = [[e + 1] * i for e, i in enumerate(cluster_biofeatures)][::-1]
        bf_clusters = [item for sublist in bf_clusters for item in sublist]
        df['cluster'] = pd.Series(bf_clusters, index=df.index)
        df = df.reset_index().set_index(['cluster'] + list(df.index.names))

        ss_clusters = [[e + 1] * i for e, i in enumerate(cluster_samplesets)]
        ss_clusters = [item for sublist in ss_clusters for item in sublist]
        df = df.T
        df['cluster'] = pd.Series(ss_clusters, index=df.index)
        df = df.reset_index().set_index(['cluster'] + list(df.index.names))
        df = df.T

        self.__modules_n__ += 1
        _name = kwargs.get('name', 'Module_{n}'.format(n=self.__modules_n__))
        return Module(compendium=self, df=df, name=_name)

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
from pycompass.graphql_query import QuerySet
from pycompass.module import Module
import pandas as pd
import numpy as np


class Compendium():
    ALLOWED_OBJECTS = {
        'experiments': 'ExperimentType',
        'dataSources': 'DataSourceType',
        'biofeatures': 'BioFeatureType',
        'sampleSets': 'SampleSetType'
    }

    def __init__(self, url, name="vespucci", version=None, database=None, normalization=None):
        self.url = url
        self.name = name
        self.version = version
        self.database = database
        self.normalization = normalization

    @property
    def available_compendium(self):
        query = '''
        {
          compendia {
            name,
            fullName,
            description
            versions {
              versionNumber,
              versionAlias,
              databases {
                name,
                normalizations
              }
            }
          }
        }'''
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
        ss = kwargs.get('samplesets', None)
        bf = kwargs.get('biofeatures', None)
        qs = QuerySet(self, 'modules')

        try:
            bf = [x.id for x in bf]
            ss = [x.id for x in ss]
        except AttributeError as e:
            pass

        qs.filter(
            biofeaturesIds=','.join(bf),
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
                }}
            }}
        }}       
        '''.format(
            compendium=self.name,
            connections=', ' + ','.join(connections),
            biofeatures=','.join('"{}"'.format(x) for x in bf),
            samplesets=','.join('"{}"'.format(x) for x in ss)
        )
        json = qs.__run_query__(query)
        bf_idx = [x['id'] for x in json['data']['plotHeatmap']['sortedBiofeatures']]
        ss_idx = [x['id'] for x in json['data']['plotHeatmap']['sortedSamplesets']]

        df = df.loc[bf_idx][ss_idx]

        return Module(compendium=self, df=df)
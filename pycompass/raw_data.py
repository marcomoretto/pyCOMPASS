from pycompass import BiologicalFeature
from pycompass.query import query_getter, run_query
import numpy as np
from pycompass.sample import Sample


class RawData:
    '''
    The RawData class wraps a Sample object to return its raw data
    '''

    def __init__(self, sample):
        self.sample = sample
        self.__values__ = np.array([])
        self.__value_types__ = np.array([])
        self.__biofeature_reporters__ = np.array([])

        query = '''{{
              rawData(compendium:"{compendium}", sampleId:"{sample_id}") {{
                values,
                valueTypes,
                biofeatureReporters
                biofeatures {{
                  edges {{
                    node {{
                      id
                    }}
                  }}
                }}
              }}
            }}'''.format(compendium=self.sample.compendium.compendium_name, sample_id=sample.id)
        json = run_query(self.sample.compendium.connection.url, query)
        raw_data = json['data']['rawData']
        self.__values__ = np.array(raw_data['values'])
        self.__value_types__ = np.array(raw_data['valueTypes'])
        self.__biofeature_reporters__ = np.array(raw_data['biofeatureReporters'])
        self.__biofeature_ids__ = [e['node']['id'] for e in raw_data['biofeatures']['edges']]

    def get_values(self):
        return self.__values__

    def get_value_types(self):
        return self.__value_types__

    def get_biofeatures(self):
        bfs = {bf.id: bf for bf in BiologicalFeature.using(self.sample.compendium).get(filter={'id_In': list(set(self.__biofeature_ids__))})}
        return [bfs[id] for id in self.__biofeature_ids__]

    def get_biofeature_reporters(self):
        return self.__biofeature_reporters__
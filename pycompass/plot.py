from pycompass.query import run_query
from pycompass.biological_feature import BiologicalFeature
from pycompass.sample_set import SampleSet


class Plot:
    '''
    The Plot class wraps a module and provides different methods to plot it
    '''

    def __init__(self, module):
        self.module = module
        self.plot_types = None

        query = '''
            {{
              plotName(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}") {{
                heatmap,
                network
              }}
            }}
        '''.format(compendium=self.module.compendium.compendium_name,
                   version=self.module.compendium.version,
                   database=self.module.compendium.database,
                   normalization=self.module.compendium.normalization)
        self.plot_types = run_query(self.module.compendium.connection.url, query)['data']['plotName']

        query = '''
                    {{
                      plotName(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}") {{
                        distribution
                      }}
                    }}
                '''.format(compendium=self.module.compendium.compendium_name,
                           version=self.module.compendium.version,
                           database=self.module.compendium.database,
                           normalization=self.module.compendium.normalization)
        pt = run_query(self.module.compendium.connection.url, query)['data']['plotName']['distribution']
        self.plot_types['distribution'] = []
        if len(pt):
            self.plot_rank_name = {}
            for p in pt:
                self.plot_types['distribution'].append(p[0])
                self.plot_rank_name[p[0]] = p[1]


    def plot_heatmap(self, plot_type=None, output_format='html', *args, **kwargs):
        '''
        Get the HTML or JSON code that plot module heatmaps

        :param plot_type: the plot type
        :param output_format: html or json
        :return: str
        '''
        if plot_type is None:
            plot_type = self.plot_types['heatmap'][0]
        if plot_type not in self.plot_types['heatmap']:
            raise Exception('Invalid plot type. Options are ' + ','.join(self.plot_types['heatmap']))
        _options = []
        for k, v in kwargs.items():
            _v = str(v)
            if type(v) == str:
                _v = '"' + str(v) + '"'
            elif type(v) == bool:
                _v = str(v).lower()
            _options.append(str(k) + ':' + _v)
        options = ',' if len(_options) > 0 else ''
        options += ','.join(_options)
        query = '''
            {{
                plotHeatmap(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}", plotType:"{plot_type}",
                biofeaturesIds:[{biofeatures}], samplesetIds:[{samplesets}] {options}) {{
                    {output},
                    sortedSamplesets {{
                      id
                    }},
                    sortedBiofeatures {{
                      id
                    }}
                }}
            }}
        '''.format(compendium=self.module.compendium.compendium_name,
                   version=self.module.compendium.version,
                   database=self.module.compendium.database,
                   normalization=self.module.compendium.normalization,
                   plot_type=plot_type,
                   output=output_format,
                   options=options,
                   biofeatures='"' + '","'.join([bf.id for bf in self.module.biological_features]) + '"',
                   samplesets='"' + '","'.join([ss.id for ss in self.module.sample_sets]) + '"')
        json = run_query(self.module.compendium.connection.url, query)
        bf = [x['id'] for x in json['data']['plotHeatmap']['sortedBiofeatures']]
        ss = [x['id'] for x in json['data']['plotHeatmap']['sortedSamplesets']]

        sorted_bf = [tuple for x in bf for tuple in self.module.biological_features if tuple.id == x]
        sorted_ss = [tuple for x in ss for tuple in self.module.sample_sets if tuple.id == x]

        return json['data']['plotHeatmap'][output_format], sorted_bf, sorted_ss

    def plot_network(self, plot_type=None, output_format='html', *args, **kwargs):
        '''
        Get the HTML or JSON code that plot the module networks

        :param plot_type: the plot type
        :param output_format: html or json
        :return: str
        '''
        if plot_type is None:
            plot_type = self.plot_types['network'][0]
        if plot_type not in self.plot_types['network']:
            raise Exception('Invalid plot type. Options are ' + ','.join(self.plot_types['network']))
        _options = []
        for k, v in kwargs.items():
            _v = str(v)
            if type(v) == str:
                _v = '"' + str(v) + '"'
            elif type(v) == bool:
                _v = str(v).lower()
            _options.append(str(k) + ':' + _v)
        options = ',' if len(_options) > 0 else ''
        options += ','.join(_options)
        query = '''
            {{
                plotNetwork(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                plotType:"{plot_type}", biofeaturesIds:[{biofeatures}],
                samplesetIds:[{samplesets}] {options}) {{
                    {output}
                }}
            }}
        '''.format(compendium=self.module.compendium.compendium_name,
                   version=self.module.compendium.version,
                   database=self.module.compendium.database,
                   normalization=self.module.compendium.normalization,
                   plot_type=plot_type,
                   output=output_format,
                   options=options,
                   biofeatures='"' + '","'.join([bf.id for bf in self.module.biological_features]) + '"',
                   samplesets='"' + '","'.join([ss.id for ss in self.module.sample_sets]) + '"')
        json = run_query(self.module.compendium.connection.url, query)
        return json['data']['plotNetwork'][output_format]

    def plot_distribution(self, plot_type, output_format='html', get_rank=False, *args, **kwargs):
        '''
        Get the HTML or JSON code that plot module distributions

        :param plot_type: the plot type
        :param output_format: html or json
        :return: str
        '''
        if plot_type is None:
            plot_type = self.plot_types['distribution'][0]
        if plot_type not in self.plot_types['distribution']:
            raise Exception('Invalid plot type. Options are ' + ','.join(self.plot_types['distribution']))
        _options = []
        for k, v in kwargs.items():
            _v = str(v)
            if type(v) == str:
                _v = '"' + str(v) + '"'
            elif type(v) == bool:
                _v = str(v).lower()
            _options.append(str(k) + ':' + _v)
        options = ',' if len(_options) > 0 else ''
        options += ','.join(_options)
        rank = '''
            , ranking
                {
                    id,
                    name,
                    value
                }
        '''
        if not get_rank:
            rank = ''
        query = '''
            {{
                plotDistribution(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                plotType:"{plot_type}", biofeaturesIds:[{biofeatures}],
                samplesetIds:[{samplesets}] {options}) {{
                    {output}
                    {rank}
                }}
            }}
        '''.format(compendium=self.module.compendium.compendium_name,
                   version=self.module.compendium.version,
                   database=self.module.compendium.database,
                   normalization=self.module.compendium.normalization,
                   plot_type=plot_type,
                   output=output_format,
                   options=options,
                   biofeatures='"' + '","'.join([bf.id for bf in self.module.biological_features]) + '"',
                   samplesets='"' + '","'.join([ss.id for ss in self.module.sample_sets]) + '"',
                   rank=rank)
        json = run_query(self.module.compendium.connection.url, query)
        if get_rank:
            return json['data']['plotDistribution'][output_format], json['data']['plotDistribution']['ranking']
        return json['data']['plotDistribution'][output_format]

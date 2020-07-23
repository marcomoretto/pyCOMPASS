from pycompass.biological_feature import BiologicalFeature
from pycompass.query import query_getter, run_query
from pycompass.sample_set import SampleSet
from pycompass.utils import get_compendium_object
import numpy as np
import pickle as pk


class Module:
    '''
    A module is a subset of the entire compendium 2D matrix that holds the quantitative values. Rows are BiologicalFeatures and columns are SampleSets
    '''

    def __init__(self, *args, **kwargs):
        self.biological_features = tuple()
        self.sample_sets = tuple()
        self.name = None
        self.id = None
        self.__normalized_values__ = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def by(self, *args, **kwargs):
        raise NotImplementedError()

    def write_to_file(self, filename):
        '''
        Dump a module into a local file

        :param filename:
        :return:
        '''
        obj = {
            'bfs': [bf.id for bf in self.biological_features],
            'sss': [ss.id for ss in self.sample_sets],
            'compendium': self.compendium,
            'values': self.values
        }
        with open(filename, 'wb') as fo:
            pk.dump(obj, fo)

    @staticmethod
    def read_from_file(filename):
        '''
        Read module data from a local file

        :param filename:
        :return:
        '''
        module = Module()
        with open(filename, 'rb') as fi:
            obj = pk.load(fi)
            if obj:
                module.compendium = obj['compendium']
                module.biological_features = BiologicalFeature.using(module.compendium).get(filter={'id_In': obj['bfs']})
                module.sample_sets = SampleSet.using(module.compendium).get(filter={'id_In': obj['sss']})
                module.__normalized_values__ = obj['values']
        return module

    def create(self, biofeatures=None, samplesets=None, rank=None, cutoff=None):
        '''
        Create a new module

        :param biofeatures: the biofeatures list for the module (inferred if None)
        :param samplesets: the samplesets list for the module (inferred if None)
        :param rank: the rank method to be used for the inference
        :param cutoff: the cutoff to be used for the inference
        :param normalization: the normalization to be used for the inference
        :return: a Module object
        '''
        _bf_limit = 50
        _ss_limit = 50
        self.biological_features = None
        self.sample_sets = None
        if biofeatures:
            self.biological_features = tuple(biofeatures)
        if samplesets:
            self.sample_sets = tuple(samplesets)
        self.name = None
        self.id = None
        # check that everything is ok to retrieve the normalized values
        if not self.biological_features and not self.sample_sets:
            raise Exception('You need to provide at least biofeatures or samplesets')
        elif self.biological_features is None:
            norm = None
            for ss in self.sample_sets:
                if ss.normalization and norm is None:
                    norm = ss.normalization
                if ss.normalization != norm:
                    raise Exception('You cannot mix SampleSets with different normalization')
            setattr(self, 'normalization', norm)
            all_ranks = self.compendium.get_score_rank_methods()['biologicalFeatures']
            _rank = rank
            if not rank:
                _rank = all_ranks[0]
            else:
                if rank not in all_ranks:
                    raise Exception('Invalid rank: choises are ' + ','.join(all_ranks))
            setattr(self, 'rank', _rank)
            # get first _bf_limit biofeatures automatically
            _bf = self.compendium.rank_biological_features(self, rank_method=_rank, cutoff=cutoff)
            _bf = _bf['ranking']['id'][:_bf_limit]
            self.biological_features = tuple(BiologicalFeature.using(self.compendium).get(
                filter={'id_In': str(_bf)}
            ))
        elif self.sample_sets is None:
            all_ranks = self.compendium.get_score_rank_methods()['sampleSets']
            _rank = rank
            if not rank:
                _rank = all_ranks[0]
            else:
                if rank not in all_ranks:
                    raise Exception('Invalid rank: choises are ' + ','.join(all_ranks))
            setattr(self, 'rank', _rank)
            # get first _ss_limit samplesets automatically
            _ss = self.compendium.rank_sample_sets(self, rank_method=_rank, cutoff=cutoff)
            _ss = _ss['ranking']['id'][:_ss_limit]
            self.sample_sets = tuple(SampleSet.using(self.compendium).get(
                filter={'id_In': str(_ss)}
            ))
        # now we biofeatures and samplesets
        setattr(self, '__normalized_values__', None)
        self.values

        return self

    @property
    def values(self):
        '''
        Get module values

        :return: np.array
        '''
        def _get_normalized_values(filter=None, fields=None):
            query = '''\
                {{\
                    {base}(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}" {filter}) {{\
                        {fields}\
                    }}\
                }}\
            '''.format(base='modules', compendium=self.compendium.compendium_name,
                       version=self.compendium.version,
                       database=self.compendium.database,
                       normalization=self.compendium.normalization,
                       filter=', biofeaturesIds:[' + ','.join(['"' + bf.id + '"' for bf in self.biological_features]) + '],' +
                            'samplesetIds: [' + ','.join(['"' + ss.id + '"' for ss in self.sample_sets]) + ']', fields=fields)
            return run_query(self.compendium.connection.url, query)

        if self.__normalized_values__ is None or len(self.__normalized_values__) == 0:
            response = _get_normalized_values(fields='''normalizedValues, biofeatures {
                          edges {
                            node {
                              id
                            }
                          }
                        },
                        sampleSets {
                          edges {
                            node {
                              id
                            } 
                          }
                        }''')
            self.__normalized_values__ = np.array(response['data']['modules']['normalizedValues'])
            _ss = [x['node']['id'] for x in response['data']['modules']['sampleSets']['edges']]
            _bf = [x['node']['id'] for x in response['data']['modules']['biofeatures']['edges']]
            self.sample_sets = {ss.id:ss for ss in SampleSet.using(self.compendium).get(
                filter={'id_In': str(_ss)}
            )}
            self.sample_sets = [self.sample_sets[i] for i in _ss]
            self.biological_features = {bf.id:bf for bf in BiologicalFeature.using(self.compendium).get(
                filter={'id_In': str(_bf)}
            )}
            self.biological_features = [self.biological_features[i] for i in _bf]
        return self.__normalized_values__

    def add_biological_features(self, biological_features=[]):
        '''
        Add biological feature to the module

        :param biological_features: list of BioFeatures objects
        :return: None
        '''
        before = set(self.biological_features)
        after = set(self.biological_features + biological_features)
        if len(set.intersection(before, after)) != 0:
            self.__normalized_values__ = None
            self.biological_features = list(after)

    def add_sample_sets(self, sample_sets=[]):
        '''
        Add sample sets to the module

        :param sample_sets: list of SampleSet objects
        :return: None
        '''
        before = set(self.sample_sets)
        after = set(self.sample_sets + sample_sets)
        if len(set.intersection(before, after)) != 0:
            self.__normalized_values__ = None
            self.sample_sets = list(after)

    def remove_biological_features(self, biological_features=[]):
        '''
        Remove biological feature from the module

        :param biological_features: list of BioFeatures objects
        :return: None
        '''
        before = set(self.biological_features)
        after = set(self.biological_features) - set(biological_features)
        if len(set.intersection(before, after)) != 0:
            self.__normalized_values__ = None
            self.biological_features = list(after)

    def remove_sample_sets(self, sample_sets=[]):
        '''
        Remove sample sets from the module

        :param sample_sets: list of SampleSet objects
        :return: None
        '''
        before = set(self.sample_sets)
        after = set(self.sample_sets) - set(sample_sets)
        if len(set.intersection(before, after)) != 0:
            self.__normalized_values__ = None
            self.sample_sets = list(after)

    @staticmethod
    def union(first, second, biological_features=True, sample_sets=True):
        '''
        Union of two modules

        :param first: first module
        :param second: second module
        :return: a new Module
        '''
        if not isinstance(first, Module) or not isinstance(second, Module):
            raise Exception('Arguments must be valid Module objects!')
        if first.compendium != second.compendium:
            raise Exception('Module objects must be from the same Compendium!')
        if first.normalization != second.normalization:
            raise Exception('Module objects must have the same normalization!')
        compendium = first.compendium
        normalization = first.normalization
        bf = set(first.biological_features)
        ss = set(first.sample_sets)
        if biological_features:
            bf = set.union(bf, set(second.biological_features))
            bf = list(bf)
        if sample_sets:
            ss = set.union(ss, set(second.sample_sets))
            ss = list(ss)
        m = Module()
        m.sample_sets = ss
        m.biological_features = bf
        m.compendium = compendium
        m.normalization = normalization
        m.rank = None
        m.values
        return m

    @staticmethod
    def intersection(first, second, biological_features=True, sample_sets=True):
        '''
        Intersection of two modules

        :param first: first module
        :param second: second module
        :return: a new Module
        '''
        if not isinstance(first, Module) or not isinstance(second, Module):
            raise Exception('Arguments must be valid Module objects!')
        if first.compendium != second.compendium:
            raise Exception('Module objects must be from the same Compendium!')
        if first.normalization != second.normalization:
            raise Exception('Module objects must have the same normalization!')
        compendium = first.compendium
        normalization = first.normalization
        bf = set(first.biological_features)
        ss = set(first.sample_sets)
        if biological_features:
            bf = set.intersection(bf, set(second.biological_features))
            bf = list(bf)
            if len(bf) == 0:
                raise Exception("There are no biological features in common between these two modules!")
        if sample_sets:
            ss = set.intersection(ss, set(second.sample_sets))
            ss = list(ss)
            if len(ss) == 0:
                raise Exception("There are no sample sets in common between these two modules!")
        m = Module()
        m.sample_sets = ss
        m.biological_features = bf
        m.compendium = compendium
        m.normalization = normalization
        m.rank = None
        m.values
        return m

    @staticmethod
    def difference(first, second, biological_features=True, sample_sets=True):
        '''
        Difference between two modules

        :param first: first module
        :param second: second module
        :return: a new Module
        '''
        if not isinstance(first, Module) or not isinstance(second, Module):
            raise Exception('Arguments must be valid Module objects!')
        if first.compendium.compendium_name != second.compendium.compendium_name:
            raise Exception('Module objects must be from the same Compendium!')
        if first.normalization != second.normalization:
            raise Exception('Module objects must have the same normalization!')
        compendium = first.compendium
        normalization = first.normalization
        bf = set([_bf.id for _bf in first.biological_features])
        ss = set([_ss.id for _ss in first.sample_sets])
        if biological_features:
            bf = set.difference(bf, set([_bf.id for _bf in second.biological_features]))
            bf = list(bf)
            if len(bf) == 0:
                raise Exception("There are no biological features in common between these two modules!")
        if sample_sets:
            ss = set.difference(ss, set([_ss.id for _ss in second.sample_sets]))
            ss = list(ss)
            if len(ss) == 0:
                raise Exception("There are no sample sets in common between these two modules!")
        m = Module()
        m.sample_sets = SampleSet.using(compendium).get(filter={'id_In': ss})
        m.biological_features = BiologicalFeature.using(compendium).get(filter={'id_In': bf})
        m.compendium = compendium
        m.normalization = normalization
        m.rank = None
        m.values
        return m

    def split_module_by_biological_features(self):
        '''
        Split the current module in different modules dividing the module in distinct groups of coexpressed biological features

        :return: list of Modules
        '''
        raise NotImplementedError()

    def split_module_by_sample_sets(self):
        '''
        Split the current module in different modules dividing the module in distinct groups of sample_sets
        showing similar values.

        :return: list of Modules
        '''
        raise NotImplementedError()

    @staticmethod
    def using(compendium):
        cls = get_compendium_object(Module)
        return cls(compendium=compendium)

from operator import itemgetter

from pycompass.biological_feature import BiologicalFeature
from pycompass.query import query_getter, run_query
from pycompass.sample_set import SampleSet
from pycompass.utils import get_compendium_object
import numpy as np


class Module:

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.biological_features = []
        self.sample_sets = []
        self.__normalized_values__ = None

    def by(self, *args, **kwargs):
        raise NotImplementedError()

    def get(self, filter=None, fields=None):
        '''
        Get biological feature

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of BiologicalFeature objects
        '''
        #@query_getter('biofeatures',
        #              ['id', 'name', 'description', 'biofeaturevaluesSet { edges { node { bioFeatureField { name }, value } } }'])
        #def _get_biological_features(obj, filter=None, fields=None):
        #    pass
        #return [BiologicalFeature(**dict({'compendium': self.compendium}, **bf))
        #        for bf in _get_biological_features(self.compendium, filter=filter, fields=fields)]
        pass

    def create(self, biofeatures=None, samplesets=None, rank=None, cutoff=None, normalization=None):
        _bf_limit = 50
        _ss_limit = 50
        self.biological_features = biofeatures
        self.sample_sets = samplesets
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
            all_ranks = self.compendium.normalization[self.normalization]['scoreRankMethods']['biologicalFeatures']
            _rank = rank
            if not rank:
                _rank = all_ranks[0]
            else:
                if rank not in all_ranks:
                    raise Exception('Invalid rank: choises are ' + ','.join(all_ranks))
            setattr(self, 'rank', _rank)
            # get first _bf_limit biofeatures automatically
            _bf = self.rank_biological_features(_rank, cutoff=cutoff)
            _bf = _bf['ranking']['id'][:_bf_limit]
            self.biological_features = BiologicalFeature.using(self.compendium).get(
                filter={'id_In': str(_bf)}
            )
        elif self.sample_sets is None:
            if normalization:
                setattr(self, 'normalization', normalization)
            else:
                setattr(self, 'normalization', list(self.compendium.normalization.keys())[0])
            all_ranks = self.compendium.normalization[self.normalization]['scoreRankMethods']['sampleSets']
            _rank = rank
            if not rank:
                _rank = all_ranks[0]
            else:
                if rank not in all_ranks:
                    raise Exception('Invalid rank: choises are ' + ','.join(all_ranks))
            setattr(self, 'rank', _rank)
            # get first _ss_limit samplesets automatically
            _ss = self.rank_sample_sets(_rank, cutoff=cutoff)
            _ss = _ss['ranking']['id'][:_ss_limit]
            self.sample_sets = SampleSet.using(self.compendium).get(
                filter={'id_In': str(_ss)}
            )
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
                    {base}(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}" {filter}) {{\
                        {fields}\
                    }}\
                }}\
            '''.format(base='modules', compendium=self.compendium.compendium_name,
                       normalization=self.normalization,
                       rank=self.rank,
                       filter=', biofeaturesIds:[' + ','.join(['"' + bf.id + '"' for bf in self.biological_features]) + '],' +
                            'samplesetIds: [' + ','.join(['"' + ss.id + '"' for ss in self.sample_sets]) + ']', fields=fields)
            return run_query(self.compendium.connection.url, query)

        if self.__normalized_values__ is None or len(self.__normalized_values__) == 0:
            response = _get_normalized_values(fields="normalizedValues")
            self.__normalized_values__ = np.array(response['data']['modules']['normalizedValues'])
        return self.__normalized_values__

    def rank_sample_sets(self, rank_method=None, cutoff=None):
        '''
        Rank all sample sets on the module's biological features using rank_method

        :param rank_method:
        :param cutoff:
        :return:
        '''
        bf = [_bf.id for _bf in self.biological_features]
        query = '''
            {{
                ranking(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}", 
                        biofeaturesIds:[{biofeatures}]) {{
                            id,
                            name,
                            value
            }}
        }}
        '''.format(compendium=self.compendium.compendium_name, normalization=self.normalization, rank=rank_method,
                   biofeatures='"' + '","'.join(bf) + '"')
        json = run_query(self.compendium.connection.url, query)
        r = json['data']
        if cutoff:
            idxs = [i for i, v in enumerate(r['ranking']['value']) if v >= cutoff]
            r['ranking']['id'] = itemgetter(*idxs)(r['ranking']['id'])
            r['ranking']['name'] = itemgetter(*idxs)(r['ranking']['name'])
            r['ranking']['value'] = itemgetter(*idxs)(r['ranking']['value'])
        return r

    def rank_biological_features(self, rank_method=None, cutoff=None):
        '''
        Rank all biological features on the module's sample set using rank_method

        :param rank_method:
        :param cutoff:
        :return:
        '''
        ss = [ss.id for ss in self.sample_sets]
        query = '''
            {{
                ranking(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}", 
                        samplesetIds:[{sample_set}]) {{
                            id,
                            name,
                            value
            }}
        }}
        '''.format(compendium=self.compendium.compendium_name, normalization=self.normalization, rank=rank_method,
                   sample_set='"' + '","'.join(ss) + '"')
        json = run_query(self.compendium.connection.url, query)
        r = json['data']
        if cutoff:
            idxs = [i for i,v in enumerate(r['ranking']['value']) if v >= cutoff]
            r['ranking']['id'] = itemgetter(*idxs)(r['ranking']['id'])
            r['ranking']['name'] = itemgetter(*idxs)(r['ranking']['name'])
            r['ranking']['value'] = itemgetter(*idxs)(r['ranking']['value'])
        return r

    @staticmethod
    def using(compendium):
        cls = get_compendium_object(Module)
        return cls(compendium=compendium)

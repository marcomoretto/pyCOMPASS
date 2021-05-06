from pycompass.sample import Sample
from pycompass.ontology_node import OntologyNode
from pycompass.biological_feature import BiologicalFeature
from pycompass.query import query_getter, run_query
from pycompass.sample_set import SampleSet
from pycompass.utils import get_compendium_object
import numpy as np
import io
import pickle as pk
import zipfile


class Module:
    '''
    A module is a subset of the entire compendium 2D matrix that holds the quantitative values. Rows are BiologicalFeatures and columns are SampleSets
    '''

    class ModuleDescription:

        class SampleGenerator(object):
            def __init__(self, gen, length):
                self.gen = gen
                self.length = length

            def __len__(self):
                return self.length

            def __iter__(self):
                return self.gen

        def __init__(self, module, response):
            self.module = module
            self.categories = {}
            nodes_original_ids = set()
            for c in response['data']['modules']['samplesDescriptionSummary']:
                nodes_original_ids.add(c['category'])
                for c in c['details']:
                    nodes_original_ids.add(c['originalId'])

            nodes_map = {o.originalId: o for o in OntologyNode.using(self.module.compendium).get(filter={'originalId_In': list(nodes_original_ids)})}

            for c in response['data']['modules']['samplesDescriptionSummary']:
                cat_node = nodes_map.get(c['category'], None)
                if cat_node:
                    self.categories[cat_node] = {}
                for d in c['details']:
                    det_node = nodes_map.get(d['originalId'], None)
                    if det_node:
                        self.categories[cat_node][det_node] = Module.ModuleDescription.SampleGenerator(self.__sample_generator__(d['samples']), len(d['samples']))

        def __sample_generator__(self, sample_ids):
            for s in Sample.using(self.module.compendium).get(filter={'id_In': [s['id'] for s in sample_ids]}):
                yield s

        def __str__(self):
            f = '{cat_name} ({cat_id}), {d_name} ({d_id}): {n_samples} samples'
            s = []
            for c, d in self.categories.items():
                for cs, ss in d.items():
                    s.append(f.format(cat_name=c.termShortName, cat_id=c.originalId, d_name=cs.termShortName, d_id=cs.originalId, n_samples=str(len(ss))))
            return '\n'.join(s)

        def __repr__(self):
            return self.__str__()

    class ModuleEnrichment:

        def __init__(self, module, response):
            self.module = module
            self.sample_categories = {}
            self.biofeature_categories = {}

            nodes_original_ids = set()
            for c in response['data']['modules']['samplesetAnnotationEnrichment']:
                for ot in c['ontologyTerm']:
                    nodes_original_ids.add(ot['ontologyId'])
            if nodes_original_ids:
                nodes_map = {o.originalId: o for o in OntologyNode.using(self.module.compendium).get(filter={'originalId_In': list(nodes_original_ids)})}
                for c in response['data']['modules']['samplesetAnnotationEnrichment']:
                    self.sample_categories[c['ontology']] = {}
                    for ot in c['ontologyTerm']:
                        node = nodes_map.get(ot['ontologyId'], None)
                        if node:
                            self.sample_categories[c['ontology']][node] = float(ot['pValue'])

            nodes_original_ids = set()
            for c in response['data']['modules']['biofeatureAnnotationEnrichment']:
                for ot in c['ontologyTerm']:
                    nodes_original_ids.add(ot['ontologyId'])
            if nodes_original_ids:
                nodes_map = {o.originalId: o for o in OntologyNode.using(self.module.compendium).get(filter={'originalId_In': list(nodes_original_ids)})}
                for c in response['data']['modules']['biofeatureAnnotationEnrichment']:
                    self.biofeature_categories[c['ontology']] = {}
                    for ot in c['ontologyTerm']:
                        node = nodes_map.get(ot['ontologyId'], None)
                        if node:
                            self.biofeature_categories[c['ontology']][node] = float(ot['pValue'])

        def __str__(self):
            f = '{bf_ss}: {cat_name}, {d_name} ({d_id}), p_value {p_value}'
            s = []
            for c, d in self.biofeature_categories.items():
                for ont, p_value in d.items():
                    s.append(f.format(bf_ss='BioFeature', cat_name=c, d_name=ont.termShortName, d_id=ont.originalId, p_value=str(p_value)))
            for c, d in self.sample_categories.items():
                for ont, p_value in d.items():
                    s.append(f.format(bf_ss='SampleSet', cat_name=c, d_name=ont.termShortName, d_id=ont.originalId, p_value=str(p_value)))
            return '\n'.join(s)

        def __repr__(self):
            return self.__str__()


    def __init__(self, *args, **kwargs):
        self.biological_features = tuple()
        self.sample_sets = tuple()
        self.name = None
        self.id = None
        self.__normalized_values__ = None
        self.__module_description__ = None
        self.__enrichment__ = None
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
            'compendium': self.compendium
        }
        meta_bytes_io = io.BytesIO()
        values_bytes_io = io.BytesIO(self.values.tobytes())
        pk.dump(obj, meta_bytes_io)
        zip_bytes_io = io.BytesIO()
        with zipfile.ZipFile(zip_bytes_io, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, data in [('__meta__', meta_bytes_io), ('__values__', values_bytes_io)]:
                zip_file.writestr(file_name, data.getvalue())

        fn = filename
        if not fn.endswith('.cmf'):
            fn += '.cmf'
        with open(fn, 'wb') as f:
            f.write(zip_bytes_io.getvalue())\

    @staticmethod
    def read_from_file(filename):
        '''
        Read module data from a local file

        :param filename:
        :return:
        '''
        archive = zipfile.ZipFile(filename, 'r')
        meta = archive.read('__meta__')
        values = archive.read('__values__')
        meta_bytes_io = io.BytesIO(meta)
        values_bytes_io = io.BytesIO(values)
        meta_obj = pk.load(meta_bytes_io)
        module = Module()
        module.compendium = meta_obj['compendium']
        bfs = {bf.id: bf for bf in BiologicalFeature.using(module.compendium).get(filter={'id_In': meta_obj['bfs']})}
        sss = {ss.id: ss for ss in SampleSet.using(module.compendium).get(filter={'id_In': meta_obj['sss']})}
        module.biological_features = [bfs[x] for x in meta_obj['bfs']]
        module.sample_sets = [sss[x] for x in meta_obj['sss']]
        module.__normalized_values__ = np.reshape(np.frombuffer(values_bytes_io.getvalue()),
                                                  (len(meta_obj['bfs']), len(meta_obj['sss'])))
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
        setattr(self, '__module_description__', None)
        setattr(self, '__enrichment__', None)
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

    def get_description(self):
        '''
        Get module brief sample description using ontology annotation terms

        :return: ModuleDescription
        '''
        def _get_samples_short_description(filter=None, fields=None):
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
                       filter=', biofeaturesIds:[' + ','.join(
                           ['"' + bf.id + '"' for bf in self.biological_features]) + '],' +
                              'samplesetIds: [' + ','.join(['"' + ss.id + '"' for ss in self.sample_sets]) + ']',
                       fields=fields)
            return run_query(self.compendium.connection.url, query)

        if self.__module_description__ is None:
            response = _get_samples_short_description(fields='''samplesDescriptionSummary {
                        category
                        details {
                            originalId
                            termShortName
                            samples {
                                id
                            }
                        }          
                }''')
            self.__module_description__ = Module.ModuleDescription(self, response)
        return self.__module_description__

    def get_enrichment(self, bf_p_value=0.05, ss_p_value=0.05):
        '''
        Get module ontology annotation terms enrichment

        :return: ModuleEnrichment
        '''

        def _get_enrichment(filter=None, fields=None):
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
                       filter=', biofeaturesIds:[' + ','.join(
                           ['"' + bf.id + '"' for bf in self.biological_features]) + '],' +
                              'samplesetIds: [' + ','.join(['"' + ss.id + '"' for ss in self.sample_sets]) + ']',
                       fields=fields)
            return run_query(self.compendium.connection.url, query)

        if self.__enrichment__ is None:
            response = _get_enrichment(fields='''samplesetAnnotationEnrichment(corrPValueCutoff: {ss_p_value}) {{\
                  ontology,\
                  ontologyTerm {{\
                    ontologyId\
                    description\
                    pValue\
                  }}\
                }}\
                biofeatureAnnotationEnrichment(corrPValueCutoff: {bf_p_value}) {{\
                  ontology\
                  ontologyTerm {{\
                    ontologyId\
                    description\
                    pValue\
                  }}\
                }}'''.format(ss_p_value=ss_p_value, bf_p_value=bf_p_value))
            self.__enrichment__ = Module.ModuleEnrichment(self, response)
        return self.__enrichment__

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

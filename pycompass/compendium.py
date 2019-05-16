from pycompass import __module
from pycompass.query import run_query, query_getter
import numpy as np

from pycompass.utils import get_factory


def new__init__(self, *args, **kwargs):
    raise ValueError('Compendium object should be created using Connect.get_compendium() or Connect.get_compendia() methods!')


class Compendium(metaclass=get_factory(new__init__)):

    def __init__(self, *args, **kwargs):
        self.compendium_name = kwargs['compendium_name']
        self.connection = kwargs['connection']
        self.compendium_full_name = kwargs['compendium_full_name']
        self.description = kwargs['description']
        self.normalization = {}
        for n in kwargs['normalization']:
            self.normalization[n] = self.__get_score_rank_methods__(n)

        return self

    def get_data_sources(self, filter=None, fields=None):
        '''
        Get the experiments data sources both local and public

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of dict
        '''
        @query_getter('dataSources', ['id', 'sourceName', 'isLocal'])
        def _get_data_sources(obj, filter=None, fields=None):
            pass
        return _get_data_sources(self, filter=filter, fields=fields)

    def get_platform_types(self, filter=None, fields=None):
        '''
        Get the platform types

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of dict
        '''
        @query_getter('platformTypes', ['id', 'name', 'description'])
        def _get_platform_types(obj, filter=None, fields=None):
            pass
        return _get_platform_types(self, filter=filter, fields=fields)

    def __get_score_rank_methods__(self, normalization):
        query = '''
            {{
              scoreRankMethods(compendium:"{compendium}", normalization:"{normalization}") {{
                sampleSets,
                biologicalFeatures
              }}
            }}
        '''.format(compendium=self.compendium_name, normalization=normalization)
        json = run_query(self.connection.url, query)
        return json['data']

    @DeprecationWarning
    @query_getter('sampleAnnoatations', ['annotation { ' +
                                            'value,' +
                                            'valueType,' +
                                            'ontologyNode {' +
                                                'originalId,' +
                                                'ontology {' +
                                                    'name } },' +
                                            'valueAnnotation {' +
                                                'ontologyNode {' +
                                                    'originalId,' +
                                                    'ontology {' +
                                                        'name }' +
                                                    'json } } }'
                                         ])
    def get_sample_annotations(self, filter=None, fields=None):
        '''
        Get ontology terms used to annotate samples

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

    @DeprecationWarning
    @query_getter('biofeatureAnnotations', ['annotationValue {' +
                                                'ontologyNode {' +
                                                    'originalId,' +
                                                        'ontology {' +
                                                            'name } } }'])
    def get_biological_feature_annotations(self, filter=None, fields=None):
        '''
        Get ontology terms used to annotate biological features

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

    @DeprecationWarning
    def get_ontology_names(self):
        '''
        Get all the available ontology names

        :param ontology_name: the ontology name
        :return:
        '''
        @query_getter('ontology', ['name'])
        def _get_ontology_hierarchy(obj, filter=None, fields=None):
            pass
        r = _get_ontology_hierarchy(self)
        return [n['node']['name'] for n in r['ontology']['edges']]

    @DeprecationWarning
    def get_ontology_hierarchy(self, ontology_name):
        '''
        Get the whole ontology structure in node-link format

        :param ontology_name: the ontology name
        :return:
        '''
        @query_getter('ontology', ['structure'])
        def _get_ontology_hierarchy(obj, filter=None, fields=None):
            pass
        return _get_ontology_hierarchy(self, filter={'name': ontology_name})

    @DeprecationWarning
    def get_samples(self, by=None, fields=None):
        '''
        Get samples by annotation_terms, experiment or name

        Example: compendium.get_samples(by={'annotation_terms': ['GROWTH.SPORULATION']})

        :param by: annotation_terms, experiment or name
        :param fields: return only specific fields
        :return: dict
        '''
        @query_getter('sampleAnnotations', ['sample {' +
                                'sampleName,' +
                                'description,' +
                                'platform {' +
                                  'platformAccessId' +
                                '}, experiment {' +
                                        'experimentAccessId' +
                                '} }'])
        def _get_samples_by_annotation(obj, filter=None, fields=None):
            pass

        @query_getter('samples', ['sampleName,' +
                                'description,' +
                                'platform {' +
                                  'platformAccessId' +
                                '}, experiment {' +
                                        'experimentAccessId' +
                                '}'])
        def _get_samples_by(obj, filter=None, fields=None):
            pass

        if 'annotation_terms' in by:
            s = _get_samples_by_annotation(self, filter={'annotationValue_OntologyNode_OriginalId_In': ','.join(by['annotation_terms'])}, fields=fields)
            return {'sample': {'edges': [{'node': n['node']['sample']} for n in s['sampleAnnotations']['edges']] } }
        if 'experiment' in by:
            return _get_samples_by(self, filter={'experiment_ExperimentAccessId': by['experiment']}, fields=fields)
        if 'name' in by:
            return _get_samples_by(self, filter={'sampleName_In': ','.join(by['name'])}, fields=fields)

    @DeprecationWarning
    def get_sample_sets(self, by=None, fields=None):
        '''
        Get sample sets by id, name or samples

        Example: compendium.get_sample_sets(by={'samples': 'GSM27218.ch1'})

        :param by: id, name or samples
        :param fields: return only specific fields
        :return: dict
        '''
        @query_getter('sampleSets', ['id,' +
                                      'name,' +
                                      'normalizationdesignsampleSet {' +
                                        'edges {' +
                                          'node {' +
                                            'sample {' +
                                              'id,' +
                                              'sampleName } } } }'])
        def _get_sample_sets_by(obj, filter=None, fields=None):
            pass

        if 'id' in by:
            return _get_sample_sets_by(self, filter={'id_In': ','.join(by['id'])})

        if 'name' in by:
            if type(by['name']) == str:
                return _get_sample_sets_by(self, filter={'name': by['name']}, fields=fields)
            elif type(by['name']) == list:
                return _get_sample_sets_by(self, filter={'name_In': ','.join(by['name'])}, fields=fields)

        if 'samples' in by:
            _samples = self.get_samples(by={'name': by['samples']}, fields=['id'])
            _ids = [s['node']['id'] for s in _samples['samples']['edges']]
            return _get_sample_sets_by(self, filter={'samples': ','.join(_ids)})

    @DeprecationWarning
    def get_biological_features(self, *args, **kwargs):
        '''
        Get biological feature by id, name or annotation_terms

        Example: compendium.get_biological_features(by={'name': 'BSU00010'})
                 compendium.get_biological_features(filter={'first': 10})

        :param by: id, name or annotation_terms
        :param fields: return only specific fields
        :param filter: return results that match only filter values
        :return: dict
        '''
        @query_getter('biofeatures', ['name,' +
                        'description,' +
                        'biofeaturevaluesSet {' +
                                'edges {' +
                                        'node {' +
                                          'bioFeatureField {' +
                                            'name' +
                                          '}, value } } }'], args, kwargs)
        def _get_biological_features_by_name(obj, args, kwargs):
            pass

        @query_getter('biofeatureAnnotations', ['bioFeature {' +
                                'name,' +
                          'description,' +
                          'biofeaturevaluesSet {' +
                            'edges {' +
                              'node {' +
                                'value,' +
                                'bioFeatureField {' +
                                  'name } } } } }'])
        def _get_biological_features_by_annotation(obj, filter=None, fields=None):
            pass

        fields = kwargs.get('fields', None)
        filter = kwargs.get('filter', {})

        if 'by' in kwargs:
            if 'id' in kwargs['by']:
                return _get_biological_features_by_name(self,
                                                        filter=dict(
                                                            {'id_In': ','.join(kwargs['by']['id'])},
                                                            **filter),
                                                        fields=fields,)
            elif 'name' in kwargs['by']:
                if type(kwargs['by']['name']) == str:
                    return _get_biological_features_by_name(self,
                                                            filter=dict(
                                                                {'name': kwargs['by']['name']},
                                                                **filter),
                                                            fields=fields)
                elif type(kwargs['by']['name']) == list:
                    return _get_biological_features_by_name(self,
                                                            filter=dict(
                                                                {'name_In': ','.join(kwargs['by']['name'])},
                                                                **filter),
                                                            fields=fields)
            elif 'annotation_terms' in kwargs['by']:
                s = _get_biological_features_by_annotation(self, filter=dict({
                            'annotationValue_OntologyNode_OriginalId_In': ','.join(kwargs['by']['annotation_terms'])},
                            **filter),
                        fields=fields)
                return {'biofeatures': {'edges': [{'node': n['node']['bioFeature']} for n in s['biofeatureAnnotations']['edges']]}}
        return _get_biological_features_by_name(self, fields=fields, filter=filter)

    def list_modules(self):
        '''
        Get the list of all saved modules for the current user and compendium

        :return: list
        '''
        @query_getter('searchModules', ['name'])
        def _list_modules(obj):
            pass
        json = _list_modules(self)
        if 'searchModules' in json and 'edges' in json['searchModules']:
            return [m['node'] for m in json['searchModules']['edges']]

    def get_module(self, name):
        '''
        Retrieve a module from the server

        :param name: the module's name
        :return: Module
        '''
        headers = {"Authorization": "JWT " + self.connection._token}
        query = '''\
                    {{\
                        {base}(compendium:"{compendium}", name:{name}) {{\
                            {fields}\
                        }}\
                    }}\
                '''.format(base='modules', compendium=self.compendium_name, name='"' + name + '"',
                           fields='normalizedValues, ' +
                            'normalization, ' +
                            'biofeatures {' +
                            'edges {' +
                            'node {' +
                            'id } } }' +
                            'sampleSets {' +
                            'edges {' +
                            'node {' +
                            'id } } }'
                           )
        json = run_query(self.connection.url, query, headers=headers)
        if 'errors' in json:
            raise Exception('Module {} does not exist'.format(name))
        bio_features = [e['node']['id'] for e in json['data']['modules']['biofeatures']['edges']]
        sample_sets = [e['node']['id'] for e in json['data']['modules']['sampleSets']['edges']]
        normalization = json['data']['modules']['normalization']
        m = self.create_module(biological_features=bio_features, sample_sets=sample_sets, normalization=normalization)
        m._name = name
        m._normalized_values = np.array(json['data']['modules']['normalizedValues'])
        return m

    def create_module(self, biological_features=[], sample_sets=[], normalization=None, rank=None):
        '''
        Create a new module object, that is a matrix with rows=biological_features and columns=sample_sets
        If only one between biological_features and sample_sets is provided, the other will be
        automatically inferred

        :param biological_features: the biological_features to be used as rows
        :param sample_sets: the sample_sets to be used as columns
        :return: Module()
        '''
        if len(sample_sets) == 0 and normalization is None:
            raise Exception('If sample_sets is empty you need to provide a normalization for the automatic retrieval of sample_sets')
        m = __module.Module.__factory_build_object__(compendium=self,
                                                     biological_features=biological_features,
                                                     sample_sets=sample_sets,
                                                     normalization=normalization,
                                                     rank=rank)
        return m


from pycompass import module
from pycompass.connect import Connect
from pycompass.query import run_query, query_getter
import numpy as np


class Compendium:

    def __init__(self, compendium_name, connection=None):
        '''
        The compendium object represent a single compendium connection

        :param compendium_name: the compendium name
        :param connection: the Connect object
        '''
        self.compendium_name = compendium_name
        self.connection = connection

    def connect(self, compendium_name, url, username=None, password=None):
        '''
        Connect to a specific compendium

        :param compendium_name: the compendium name
        :param url: the COMPASS GraphQL endpoint URL
        :param username: the username
        :param password: the password
        '''
        self.compendium_name = compendium_name
        self.connection = Connect(url, username=username, password=password)

    @query_getter('dataSources', ['id', 'sourceName', 'isLocal'])
    def get_data_sources(self, filter=None, fields=None):
        '''
        Get the experiments data sources both local and public

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

    @query_getter('platforms', ['platformAccessId', 'platformName', 'description',
                                'dataSource { sourceName }',
                                'platformType { name }'])
    def get_platforms(self, filter=None, fields=None):
        '''
        Get the experiment platforms

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

    @query_getter('platformTypes', ['id', 'name', 'description'])
    def get_platform_types(self, filter=None, fields=None):
        '''
        Get the platform types

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

    @query_getter('experiments', ['id', 'organism', 'experimentAccessId', 'experimentName', 'scientificPaperRef',
                                  'description', 'comments', 'dataSource { sourceName }'])
    def get_experiments(self, filter=None, fields=None):
        '''
        Get compendium experiments

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: dict
        '''
        pass

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

    def get_biological_features(self, by=None, fields=None):
        '''
        Get biological feature by id, name or annotation_terms

        Example: compendium.get_biological_features(by={'name': 'BSU00010'})

        :param by: name or annotation_terms
        :param fields: return only specific fields
        :return: dict
        '''
        @query_getter('biofeatures', ['name,' +
                        'description,' +
                        'biofeaturevaluesSet {' +
                                'edges {' +
                                        'node {' +
                                          'bioFeatureField {' +
                                            'name' +
                                          '}, value } } }'])
        def _get_biological_features_by_name(obj, filter=None, fields=None):
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

        if 'id' in by:
            return _get_biological_features_by_name(self, filter={'id_In': ','.join(by['id'])}, fields=fields)

        if 'name' in by:
            if type(by['name']) == str:
                return _get_biological_features_by_name(self, filter={'name': by['name']}, fields=fields)
            elif type(by['name']) == list:
                return _get_biological_features_by_name(self, filter={'name_In': ','.join(by['name'])}, fields=fields)


        if 'annotation_terms':
            s = _get_biological_features_by_annotation(self, filter={
                'annotationValue_OntologyNode_OriginalId_In': ','.join(by['annotation_terms'])})
            return {'biofeatures': {'edges': [{'node': n['node']['bioFeature']} for n in s['biofeatureAnnotations']['edges']]}}

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
        m = module.Module.__factory_build_object__(compendium=self,
                                                   biological_features=biological_features,
                                                   sample_sets=sample_sets,
                                                   normalization=normalization,
                                                   rank=rank)
        return m


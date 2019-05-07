import numpy as np
from operator import itemgetter

from pycompass.plot import Plot
from pycompass.query import run_query


def new__init__(self, *args, **kwargs):
    raise ValueError('Module object should be created using Compendium.create_module() method!')


class Meta(type):
    def __new__(cls, name, bases, namespace):
        old_init = namespace.get('__init__')
        namespace['__init__'] = new__init__

        def __factory_build_object__(cls_obj, *args, **kwargs):
            obj = cls_obj.__new__(cls_obj, *args, **kwargs)
            ins = old_init(obj, *args, **kwargs)
            return ins

        namespace['__factory_build_object__'] = classmethod(__factory_build_object__)
        return super().__new__(cls, name, bases, namespace)


class Module(Plot, metaclass=Meta):

    def __init__(self, *args, **kwargs):
        self._id = None
        self._name = None
        self._compendium = kwargs['compendium']
        self._normalization = kwargs['normalization']
        self._rank = kwargs['rank']
        self._biological_features = tuple(kwargs['biological_features'])
        self._sample_sets = tuple(kwargs['sample_sets'])

        self._normalized_values = None

        return self

    def _get_normalized_values(self, filter=None, fields=None):
        _fields = fields if fields is not None else "normalizedValues"
        if self._normalization is not None:
            query = '''\
                {{\
                    {base}(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}" {filter}) {{\
                        {fields}\
                    }}\
                }}\
            '''.format(base='modules', compendium=self._compendium.compendium_name,
                       normalization=self._normalization,
                       rank=self._rank,
                       filter=', biofeaturesIds:[' + ','.join(['"' + x + '"' for x in self._biological_features]) + '],' +
                            'samplesetIds: [' + ','.join(['"' + x + '"' for x in self._sample_sets]) + ']', fields=_fields)
        else:
            query = '''\
                {{\
                    {base}(compendium:"{compendium}", rank:"{rank}" {filter}) {{\
                        {fields}\
                    }}\
                }}\
            '''.format(base='modules', compendium=self._compendium.compendium_name,
                       rank=self._rank,
                       filter=', biofeaturesIds:[' + ','.join(['"' + x + '"' for x in self._biological_features]) + '],' +
                            'samplesetIds: [' + ','.join(['"' + x + '"' for x in self._sample_sets]) + ']', fields=_fields)
        return run_query(self._compendium.connection.url, query)

    @property
    def values(self):
        '''
        Get module values

        :return: np.array
        '''
        if self._normalized_values is None or len(self._normalized_values) == 0:
            fields = "normalizedValues, " + \
                 "sampleSets {" + \
                    "edges {" + \
                        "node {" + \
                            "id," + \
                 "} } }" + \
                 "biofeatures {" + \
                    "edges {" + \
                        "node {" + \
                            "id" + \
                 "} } }"
            response = self._get_normalized_values(fields=fields)
            self._normalized_values = np.array(response['data']['modules']['normalizedValues'])
            self._biological_features = [n['node']['id'] for n in response['data']['modules']['biofeatures']['edges']]
            self._sample_sets = [n['node']['id'] for n in response['data']['modules']['sampleSets']['edges']]
        return self._normalized_values

    @property
    def biological_features(self):
        if self._biological_features is not None and len(self._biological_features) > 0:
            return self._compendium.get_biological_features(by={'id': self._biological_features})
        return None

    @property
    def sample_sets(self):
        if self._sample_sets is not None and len(self._sample_sets) > 0:
            return self._compendium.get_sample_sets(by={'id': self._sample_sets})
        return None

    @property
    def name(self):
        return self._name

    def update_name(self, new_name):
        '''
        Update current module's name

        :return: boolean
        '''
        headers = {"Authorization": "JWT " + self._compendium.connection._token}
        query = '''\
                    mutation {{\
                        {base}(compendium:"{compendium}", oldName:"{old_name}", newName:"{new_name}") {{\
                            ok\
                        }}\
                    }}\
        '''.format(base='updateModuleName', compendium=self._compendium.compendium_name,
                   old_name=self._name,
                   new_name=new_name,
                   fields='ok'
                   )
        run_query(self._compendium.connection.url, query, headers=headers)
        self._name = new_name
        return True

    def delete(self):
        '''
        Delete current module from the server

        :return: boolean
        '''
        headers = {"Authorization": "JWT " + self._compendium.connection._token}
        query = '''\
                    mutation {{\
                        {base}(compendium:"{compendium}", name:"{name}") {{\
                            ok\
                        }}\
                    }}\
        '''.format(base='deleteModule', compendium=self._compendium.compendium_name,
                   name=self._name,
                   fields='ok'
                   )
        run_query(self._compendium.connection.url, query, headers=headers)
        self._name = None
        self._biological_features = None
        self._sample_sets = None

        self._normalized_values = None
        return True

    def save(self, name=None):
        '''
        Save a module on the server

        :param name: the module name
        :return: boolean
        '''
        if name is not None:
            self._name = name
        headers = {"Authorization": "JWT " + self._compendium.connection._token}
        query = '''\
            mutation {{\
                {base}(compendium:"{compendium}", name:{name}, biofeaturesIds:[{biofeaturesIds}], samplesetIds:[{samplesetIds}]) {{\
                    {fields}\
                }}\
            }}\
        '''.format(base='saveModule', compendium=self._compendium.compendium_name,
                   name='"' + self._name + '"',
                   biofeaturesIds=','.join(['"' + bf + '"' for bf in self._biological_features]),
                   samplesetIds=','.join(['"' + ss + '"' for ss in self._sample_sets]),
                   fields='ok, id'
                   )
        json = run_query(self._compendium.connection.url, query, headers=headers)
        self._id = json['data']['saveModule']['id']
        return True

    def rank_biological_features(self, rank_method=None, cutoff=None):
        '''
        Rank all biological features on the module's sample set using rank_method

        :param rank_method:
        :param cutoff:
        :return:
        '''
        headers = {"Authorization": "JWT " + self._compendium.connection._token}
        query = '''
            {{
                ranking(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}", 
                        samplesetIds:[{sample_set}]) {{
                            id,
                            name,
                            value
            }}
        }}
        '''.format(compendium=self._compendium.compendium_name, normalization=self._normalization, rank=rank_method,
                   sample_set='"' + '","'.join(self._sample_sets) + '"')
        json = run_query(self._compendium.connection.url, query, headers=headers)
        r = json['data']
        if cutoff:
            idxs = [i for i,v in enumerate(r['ranking']['value']) if v >= cutoff]
            r['ranking']['id'] = itemgetter(*idxs)(r['ranking']['id'])
            r['ranking']['name'] = itemgetter(*idxs)(r['ranking']['name'])
            r['ranking']['value'] = itemgetter(*idxs)(r['ranking']['value'])
        return r

    def add_biological_features(self, name=None):
        '''
        Add biological feature to the module using name

        Example: module.add_biological_features(name='BSU00010')

        :param name: string or list of biofeature names
        :return:
        '''
        ids = []
        if name:
            if type(name) == str:
                ids.append(
                    self._compendium.get_biological_features(
                        by={'name': name}, fields={'id'}
                    )['biofeatures']['edges'][0]['node']['id']
                )
            elif type(name) == list:
                for gname in name:
                    ids.append(
                        self._compendium.get_biological_features(
                            by={'name': gname}, fields={'id'}
                        )['biofeatures']['edges'][0]['node']['id']
                    )
        before = set(self._biological_features)
        after = set(self._biological_features + ids)
        if len(set.intersection(before, after)) != 0:
            self._normalized_values = None
            self._biological_features = list(after)

    def remove_biological_features(self, name=None):
        '''
        Remove biological feature to the module using name

        Example: module.remove_biological_features(name='BSU00010')

        :param name: str or list of biofeature names
        :return: None
        '''
        ids = []
        if name:
            if type(name) == str:
                ids.append(
                    self._compendium.get_biological_features(
                        by={'name': name}, fields={'id'}
                    )['biofeatures']['edges'][0]['node']['id']
                )
            elif type(name) == list:
                for gname in name:
                    ids.append(
                        self._compendium.get_biological_features(
                            by={'name': gname}, fields={'id'}
                        )['biofeatures']['edges'][0]['node']['id']
                    )
        before = set(self._biological_features)
        after = set(self._biological_features) - set(ids)
        if len(set.intersection(before, after)) != 0:
            self._normalized_values = None
            self._biological_features = list(after)

    def rank_sample_sets(self, rank_method=None, cutoff=None):
        '''
        Rank all sample sets on the module's biological features using rank_method

        :param rank_method:
        :param cutoff:
        :return:
        '''
        headers = {"Authorization": "JWT " + self._compendium.connection._token}
        query = '''
            {{
                ranking(compendium:"{compendium}", normalization:"{normalization}", rank:"{rank}", 
                        biofeaturesIds:[{biofeatures}]) {{
                            id,
                            name,
                            value
            }}
        }}
        '''.format(compendium=self._compendium.compendium_name, normalization=self._normalization, rank=rank_method,
                   biofeatures='"' + '","'.join(self._biological_features) + '"')
        json = run_query(self._compendium.connection.url, query, headers=headers)
        r = json['data']
        if cutoff:
            idxs = [i for i, v in enumerate(r['ranking']['value']) if v >= cutoff]
            r['ranking']['id'] = itemgetter(*idxs)(r['ranking']['id'])
            r['ranking']['name'] = itemgetter(*idxs)(r['ranking']['name'])
            r['ranking']['value'] = itemgetter(*idxs)(r['ranking']['value'])
        return r

    def add_sample_sets(self, name=None):
        '''
        Add sample sets to the module using name

        :param name: str or list of sample sets
        :return: None
        '''
        ids = []
        if name:
            if type(name) == str:
                ids.append(
                    self._compendium.get_sample_sets(
                        by={'name': name}, fields={'id'}
                    )['sampleSets']['edges'][0]['node']['id']
                )
            elif type(name) == list:
                for gname in name:
                    ids.append(
                        self._compendium.get_sample_sets(
                            by={'name': gname}, fields={'id'}
                        )['sampleSets']['edges'][0]['node']['id']
                    )
        before = set(self._sample_sets)
        after = set(self._sample_sets + ids)
        if len(set.intersection(before, after)) != 0:
            self._normalized_values = None
            self._sample_sets = list(after)

    def remove_sample_sets(self, name=None):
        '''
        Remove sample sets to the module using name

        :param name: str or list of sample sets
        :return: None
        '''
        ids = []
        if name:
            if type(name) == str:
                ids.append(
                    self._compendium.get_sample_sets(
                        by={'name': name}, fields={'id'}
                    )['sampleSets']['edges'][0]['node']['id']
                )
            elif type(name) == list:
                for gname in name:
                    ids.append(
                        self._compendium.get_sample_sets(
                            by={'name': gname}, fields={'id'}
                        )['sampleSets']['edges'][0]['node']['id']
                    )
        before = set(self._sample_sets)
        after = set(self._sample_sets) - set(ids)
        if len(set.intersection(before, after)) != 0:
            self._normalized_values = None
            self._sample_sets = list(after)

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
    def union(first, second, biological_features=True, sample_sets=True):
        '''
        Union of two modules

        :param first:
        :param second:
        :return: Module
        '''
        if type(first) != type(second):
            raise Exception('Arguments must be of the same type!')
        if type(first) != Module:
            raise Exception('Arguments must be valid Module objects!')
        if first._compendium != second._compendium:
            raise Exception('Module objects must be from the same Compendium!')
        compendium = first._compendium
        bf = set(first._biological_features)
        ss = set(first._sample_sets)
        if biological_features:
            bf = set.union(bf, set(second._biological_features))
            bf = list(bf)
        if sample_sets:
            ss = set.union(ss, set(second._sample_sets))
            ss = list(ss)
        return compendium.create_module(biological_features=bf, sample_sets=ss)

    @staticmethod
    def intersection(first, second, biological_features=True, sample_sets=True):
        '''
        Intersection of two modules

        :param first:
        :param second:
        :return: Module
        '''
        if type(first) != type(second):
            raise Exception('Arguments must be of the same type!')
        if type(first) != Module:
            raise Exception('Arguments must be valid Module objects!')
        if first._compendium != second._compendium:
            raise Exception('Module objects must be from the same Compendium!')
        compendium = first._compendium
        bf = set(first._biological_features)
        ss = set(first._sample_sets)
        if biological_features:
            bf = set.intersection(bf, set(second._biological_features))
            bf = list(bf)
        if sample_sets:
            ss = set.intersection(ss, set(second._sample_sets))
            ss = list(ss)
        return compendium.create_module(biological_features=bf, sample_sets=ss)

    @staticmethod
    def difference(first, second, biological_features=True, sample_sets=True):
        '''
        Difference between two modules

        :param first:
        :param second:
        :return: Module
        '''
        if type(first) != type(second):
            raise Exception('Arguments must be of the same type!')
        if type(first) != Module:
            raise Exception('Arguments must be valid Module objects!')
        if first._compendium != second._compendium:
            raise Exception('Module objects must be from the same Compendium!')
        compendium = first._compendium
        bf = set(first._biological_features)
        ss = set(first._sample_sets)
        if biological_features:
            bf = set.difference(bf, set(second._biological_features))
            bf = list(bf)
        if sample_sets:
            ss = set.difference(ss, set(second._sample_sets))
            ss = list(ss)
        return compendium.create_module(biological_features=bf, sample_sets=ss)

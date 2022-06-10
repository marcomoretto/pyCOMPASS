import requests
import sys
from collections import namedtuple


class QuerySet():

    class QuerySetIterator:
        _qs = []
        _index = 0

        def __init__(self, qs):
            self._qs = qs

        def __next__(self):
            if self._index < len(self._qs):
                r = self._qs[self._index]
                self._index += 1
                return r
            raise StopIteration

    def __init__(self, compendium, object_type, annotation_class=None):
        self._compendium = compendium
        self._object_type = object_type
        self._filter = {}
        self._annotation_filter = {}
        self._fields = {'id'}
        self._object_fields = {}
        self._annotation_class = annotation_class

    def __iter__(self):
        # sparql filtered ids
        filtered_ids = []
        for k, v in self._annotation_filter.items():
            filtered_ids.append(set(self._annotation_class.filter(k, v, graphql_query=self)))
        if filtered_ids:
            self._filter['id_In'] = list(set.intersection(*filtered_ids))
        # putative sparql fields
        sparql_fields = self._fields.intersection(set(self._annotation_class.QUERIES.keys()))
        # remove non-graphql fields
        try:
            self._fields = self._fields.intersection(set(self.__graphql_fields__))
            self._fields = self._fields - set(self._annotation_class.QUERIES.keys())
            # add name to the fields if there are also sparql fields (required by sparql)
            if sparql_fields:
                self._fields.add(self._annotation_class.REQUIRED_FIELD)
        except Exception as e:
            pass
        query = self.__create_query__()
        json = self.__run_query__(query)
        nodes = [x['node'] for x in json['data'][self._object_type]['edges']]

        # add sparql fields if necessary
        if sparql_fields and self._annotation_class:
            self._annotation_class(sparql_fields, nodes, compendium=self._compendium)

        # create objects
        tuples = self.__create_namedtuples__(nodes)

        return QuerySet.QuerySetIterator(tuples)

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def filter(self, *args, **kwargs):
        try:
            annotation_filter = list(self._compendium.ANNOTATION_CLASSES[self._object_type].QUERIES.keys())
        except Exception as e:
            annotation_filter = []
        for k, v in kwargs.items():
            if k in annotation_filter:
                if k not in self._annotation_filter:
                    self._annotation_filter[k] = []
                self._annotation_filter[k].append(v)
            else:
                if k not in self._filter:
                    self._filter[k] = []
                self._filter[k].append(v)
        return self

    @property
    def all_fields(self):
        if not self._object_type:
            raise Exception('No query object')
        if self._annotation_class:
            return list(set(self.__graphql_fields__ + list(self._annotation_class.QUERIES.keys())))
        return self.__graphql_fields__

    @property
    def __graphql_fields__(self):
        if not self._object_type:
            raise Exception('No query object')
        return list(set(self.__meta_type__(self._compendium.ALLOWED_OBJECTS[self._object_type])))

    def fields(self, *args, **kwargs):
        if not self._object_type:
            raise Exception('No query object')
        for field in args:
            if type(field) == str:
                self._fields.add(field)
            elif type(field) == dict:
                self._object_fields = dict(field)
        return self

    def __namedtuple_from_mapping__(self, mapping, name="Tupperware"):
        this_namedtuple_maker = namedtuple(name, mapping.keys())
        return this_namedtuple_maker(**mapping)

    def __tupperware__(self, mapping, name):
        if type(mapping) == dict:
            for k, v in mapping.items():
                mapping[k] = self.__tupperware__(v, k)
            return self.__namedtuple_from_mapping__(mapping, name)
        return mapping

    def __create_namedtuples__(self, objs):
        name = self._object_type
        obj_list = []
        for o in objs:
            obj_list.append(self.__tupperware__(o, name))
        return obj_list

    def __meta_type__(self, object_type):
        query = '''
        {{
          __type(name:"{object_type}") {{
            fields {{
              name
            }}
          }}
        }}
        '''.format(object_type=object_type)
        json = self.__run_query__(query)
        return [o['name'] for o in json['data']['__type']['fields']]

    def __create_query_module__(self):
        filter = []
        for k, v in self._filter.items():
            filter.append(k + ':' + '[' + ','.join('"{}"'.format(y) for x in v for y in x.split(',')) + ']')
        filter = ','.join(filter)
        connections = []
        if self._compendium.version:
            connections.append('version:"{}"'.format(self._compendium.version))
        if self._compendium.database:
            connections.append('database:"{}"'.format(self._compendium.database))
        if self._compendium.normalization:
            connections.append('normalization:"{}"'.format(self._compendium.normalization))
        query = '''
                {{
                  {object_type}(compendium:"{compendium}" {connections} {filter}) {{
                    normalizedValues,
                    biofeatures {{
                      edges {{
                        node {{
                          id
                        }}
                      }}
                    }},
                    sampleSets {{
                      edges {{
                        node {{
                          id
                        }} 
                      }}
                    }}
                  }}
                }}        
                '''.format(
            object_type=self._object_type,
            filter=', ' + filter,
            compendium=self._compendium.name,
            connections=', ' + ','.join(connections)
        )
        return query

    def __create_query__(self):
        filter = []
        for k, v in self._filter.items():
            _v = []
            _is_num = False
            _is_list = False
            for i in v:
                if type(i) == list:
                    if k.endswith('_In'):
                        _v.append(','.join(i))
                    else:
                        _v.append(','.join(['"{}"'.format(x) for x in i]))
                        _is_list = True
                    continue
                elif type(i) == float or type(i) == int or i.isnumeric():
                    _is_num = True
                _v.append(str(i))
            if _is_num:
                filter.append('{}:{}'.format(k, ','.join(_v)))
            elif _is_list:
                filter.append('{}:[{}]'.format(k, ','.join(_v)))
            else:
                filter.append('{}:"{}"'.format(k, ','.join(_v)))
        filter = ','.join(filter)
        connections = []
        if self._compendium.version:
            connections.append('version:"{}"'.format(self._compendium.version))
        if self._compendium.database:
            connections.append('database:"{}"'.format(self._compendium.database))
        if self._compendium.normalization:
            connections.append('normalization:"{}"'.format(self._compendium.normalization))
        query = '''
        {{
          {object_type}(compendium:"{compendium}" {connections} {filter}) {{
            edges {{
              node {{
                {fields}
                {object_fields}
              }}
            }}
          }}
        }}        
        '''.format(
            object_type=self._object_type,
            fields=','.join(self._fields),
            object_fields=str(self._object_fields).replace("'", "").replace(':', '')[1:-1],
            filter=', ' + filter,
            compendium=self._compendium.name,
            connections=', ' + ','.join(connections)
        )
        return query

    def __run_query__(self, query, headers=None, show_query=False):
        if headers:
            request = requests.post(self._compendium.url, json={'query': query}, headers=headers, verify=False)
        else:
            request = requests.post(self._compendium.url, json={'query': query}, verify=False)
        if request.status_code == 200:
            json = request.json()
            if 'errors' in json:
                raise Exception(json['errors'])
            if show_query:
                sys.stderr.write("**** GRAPHQL QUERY BEGIN ***\n")
                sys.stderr.write(query + "\n")
                sys.stderr.write("**** GRAPHQL QUERY END ***\n")
            return json
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))
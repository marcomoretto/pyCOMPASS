import requests
from functools import partial, wraps
import sys

__SHOW_GRAPHQL_QUERY__ = False

def run_query(url, query, headers=None):
    global __SHOW_GRAPHQL_QUERY__
    if headers:
        request = requests.post(url, json={'query': query}, headers=headers, verify=False)
    else:
        request = requests.post(url, json={'query': query}, verify=False)
    if request.status_code == 200:
        json = request.json()
        if 'errors' in json:
            raise Exception(json['errors'])
        if __SHOW_GRAPHQL_QUERY__:
            sys.stderr.write("**** GRAPHQL QUERY BEGIN ***\n")
            sys.stderr.write(query + "\n")
            sys.stderr.write("**** GRAPHQL QUERY END ***\n")
        return json
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


def query_getter(base, default_fields, *args_, **kwargs_):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            filter_string = ''
            if 'filter' in kwargs and kwargs['filter']:
                flt = []
                for k, v in kwargs['filter'].items():
                    if type(v) == str:
                        flt.append(k + ':"' + v + '"')
                    elif type(v) == bool:
                        flt.append(k + ':' + str(v).lower())
                    else:
                        if k.endswith('_In'):
                            flt.append(k + ':"' + ','.join([str(e) for e in v]) + '"')
                            flt[-1] = flt[-1].replace("'", '')
                        else:
                            flt.append(k + ':' + str(v).replace("'", '"'))
                filter_string = ',' + ' '.join(flt)
            if 'fields' in kwargs and kwargs['fields']:
                if type(kwargs['fields']) != list:
                    raise Exception('fields must be a list')
                fields_string = ','.join(list(set(['id'] + kwargs['fields'])))
            else:
                fields_string = ','.join(default_fields)
            query = '''\
                        {{\
                            {base}(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}" {filter}) {{\
                                edges {{\
                                    node {{\
                                        {fields}\
                                    }}\
                                }}\
                            }}\
                        }}\
                    '''.format(base=base,
                               compendium=self.compendium_name,
                               version=self.version,
                               database=self.database,
                               normalization=self.normalization,
                               filter=filter_string,
                               fields=fields_string)
            json = run_query(self.connection.url, query)
            if 'errors' in json:
                raise ValueError(json['errors'])
            return [e['node'] for e in json['data'][base]['edges']]
        return wrapper
    return actual_decorator


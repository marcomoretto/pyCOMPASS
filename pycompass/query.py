import requests
from functools import partial, wraps


def run_query(url, query, headers=None):
    if headers:
        request = requests.post(url, json={'query': query}, headers=headers)
    else:
        request = requests.post(url, json={'query': query})
    if request.status_code == 200:
        json = request.json()
        if 'errors' in json:
            raise Exception(json['errors'])
        return json
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


def query_getter(base, default_fields):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            headers = {"Authorization": "JWT " + self.connection._token}
            filter_string = ''
            if 'filter' in kwargs and kwargs['filter']:
                flt = []
                for k, v in kwargs['filter'].items():
                    if type(v) == str:
                        flt.append(k + ':"' + v + '"')
                    else:
                        flt.append(k + ':' + str(v))
                filter_string = ',' + ' '.join(flt)
            if 'fields' in kwargs and kwargs['fields']:
                fields_string = ','.join(kwargs['fields'])
            else:
                fields_string = ','.join(default_fields)
            query = '''\
                        {{\
                            {base}(compendium:"{compendium}" {filter}) {{\
                                edges {{\
                                    node {{\
                                        {fields}\
                                    }}\
                                }}\
                            }}\
                        }}\
                    '''.format(base=base, compendium=self.compendium_name, filter=filter_string, fields=fields_string)
            json = run_query(self.connection.url, query, headers=headers)
            if 'errors' in json:
                raise ValueError(json['errors'])
            return json['data']
        return wrapper
    return actual_decorator


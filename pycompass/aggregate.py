from pycompass.query import run_query


class Aggregate:

    def __init__(self, compendium, cls):
        self.compendium = compendium
        self.cls = cls

    def total_count(self):
        aggregate_fun = 'totalCount'
        query = '''\
                {{\
                    {base}(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}") {{\
                        {aggregate_fun}
                    }}\
                }}\
                '''.format(base=self.cls,
                           compendium=self.compendium.compendium_name,
                           version=self.compendium.version,
                           database=self.compendium.database,
                           normalization=self.compendium.normalization,
                           aggregate_fun=aggregate_fun
                           )
        json = run_query(self.compendium.connection.url, query)
        if 'errors' in json:
            raise ValueError(json['errors'])
        return json['data'][self.cls][aggregate_fun]
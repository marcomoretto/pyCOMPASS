from pycompass.experiment import Experiment
from pycompass.platform import Platform
from pycompass.query import query_getter, run_query
from pycompass.utils import get_compendium_object


class Sparql:

    def __init__(self, *args, **kwargs):
        pass

    def execute_query(self, query, target):
        query = '''{{
                    sparql(compendium:"{compendium}", version:"{version}", database:"{database}", normalization:"{normalization}",
                        query:"{query}", target:"{target}") {{
                        rdfTriples
                  }}
                }}'''.format(compendium=self.compendium.compendium_name,
                             version=self.compendium.version,
                             database=self.compendium.database,
                             normalization=self.compendium.normalization,
                             query=query,
                             target=target)
        json = run_query(self.compendium.connection.url, query)
        return json['data']['sparql']['rdfTriples']

    @staticmethod
    def using(compendium):
        cls = get_compendium_object(Sparql)
        return cls(compendium=compendium)
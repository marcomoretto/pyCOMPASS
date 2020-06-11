import pycompass
from pycompass.query import run_query


class Connect:
    '''
    Connect class is used to get a connection to a valid COMPASS GraphQL endpoint.
    '''

    def __init__(self, url):
        self.url = url

    def get_compass_version(self):
        '''
        Get current backend version

        :return: String
        '''
        query = '''{
            version
        }'''
        json = run_query(self.url, query)
        return json['data']['version']

    def describe_compendia(self):
        '''
        Get all available compendia

        :return: dict of compendia structure
        '''
        query = '''{
                    compendia {
                        name,
                        fullName,
                        defaultVersion,
                        description,
                        versions {
                            versionNumber,
                            versionAlias,
                            defaultDatabase,
                            databases {
                                name,
                                normalizations
                            }
                        }
                      }
                }'''
        json = run_query(self.url, query)
        return json['data']

    def get_compendium(self, name, version=None, database=None, normalization=None):
        '''
        Get a compendium by a given name, None otherwise

        :param name: the compendium name
        :param version: the compendium version (use default if None)
        :param database: the compendium database (use default if None)
        :param normalization: the compendium normalization (use default if None)
        :return: Compendium object
        '''
        query = '''{
            compendia {
                name,
                fullName,
                defaultVersion,
                description,
                versions {
                    versionNumber,
                    versionAlias,
                    defaultDatabase,
                    databases {
                        name,
                        normalizations
                    }
                }
              }
        }'''
        json = run_query(self.url, query)
        for c in json['data']['compendia']:
            _version = None
            _version_alias = None
            _database = None
            _normalization = None
            for v in c['versions']:
                if version == str(v['versionNumber']) or version == str(v['versionAlias']):
                    _version = str(v['versionNumber'])
                    _version_alias = str(v['versionAlias'])
                elif str(v['versionNumber']) == str(c['defaultVersion']) and version is None:
                    _version = str(v['versionNumber'])
                    _version_alias = str(v['versionAlias'])
                else:
                    continue
                if database is None:
                    _database = v['defaultDatabase']
                else:
                    _database = database
                for d in v['databases']:
                    if d['name'] == _database:
                        if normalization is None:
                            for n in d['normalizations']:
                                _normalization = n.replace('(default)', '').strip()
                        else:
                            _normalization = normalization

            comp = pycompass.Compendium.__factory_build_object__(
                compendium_name=c['name'],
                compendium_full_name=c['fullName'],
                description=c['description'],
                version=_version,
                version_alias=_version_alias,
                database=_database,
                normalization=_normalization,
                connection=self,
            )
            return comp

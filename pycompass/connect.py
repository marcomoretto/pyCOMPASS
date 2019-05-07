from pycompass.query import run_query


class Connect:

    def __init__(self, url, username=None, password=None):
        '''
        Connect class is used to get a connection to a valid COMPASS GraphQL endpoint. If username and password are
        provided, it will be possible to store and manage Modules on the server

        :param url: the COMPASS GraphQL endpoint URL
        :param username: the username
        :param password: the password
        '''

        self.url = url
        self._token = None
        self.username = username
        self.password = password
        self.login(username, password)

    def login(self, username=None, password=None):
        '''
        Login

        :param username:
        :param password:
        :return:
        '''

        self.username = username
        self.password = password
        if self.username and self.password:
            self._token = self._get_token()

    def signup(self, username=None, email=None, password=None):
        '''
        Signup as new user

        :param username: the username
        :param email: the user email
        :param password: the password
        :return: ok if the user has been added
        '''
        query = '''\
            mutation {{\
                {base}(username:"{username}", email:"{email}", password:"{password}") {{\
                    {fields}\
                }}\
            }}\
        '''.format(base='signup',
                   username=username,
                   email=email,
                   password=password,
                   fields='ok'
                   )
        json = run_query(self.url, query)
        self.login(username, password)
        return True

    def get_compendia(self):
        '''

        :return: list of all available compendia
        '''

        query = '''{
          compendia {
            name,
            fullName,
            description,
            normalization
          }
        }'''
        json = run_query(self.url, query)
        return json['data']

    def get_score_rank_methods(self, compendium, normalization):
        '''

        :return: list of all available score rank methods
        '''
        query = '''
            {{
              scoreRankMethods(compendium:"{compendium}", normalization:"{normalization}") {{
                sampleSets,
                biologicalFeatures
              }}
            }}
        '''.format(compendium=compendium, normalization=normalization)
        json = run_query(self.url, query)
        return json['data']

    def get_plot_type(self):
        '''

        :return: list of all available plot type
        '''
        query = "{ plotType }"
        json = run_query(self.url, query)
        return json['data']

    def _get_token(self):
        query = '''\
            mutation {{\
                tokenAuth(username: "{}", password: "{}") {{\
                    token\
                }}\
            }}\
        '''.format(self.username, self.password)
        json = run_query(self.url, query)
        return json['data']['tokenAuth']['token']

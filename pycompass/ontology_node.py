from pycompass.ontology import Ontology
from pycompass.query import query_getter
from pycompass.utils import get_compendium_object
import json


class OntologyNode:
    '''
    Ontology node class represent the different ontologie nodes used to annotate BiologicalFeature and Sample objects
    '''

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if k == 'ontology':
                self.__ontology_id__ = v['id']
            else:
                setattr(self, k, v)

    @property
    def ontology(self):
        return Ontology.using(self.compendium).get(filter={'id': self.__ontology_id__})[0]

    def by(self, *args, **kwargs):
        raise NotImplementedError()

    def get(self, filter=None, fields=None):
        '''
        Get the ontology node used to annotate samples and biological features

        :param filter: return results that match only filter values
        :param fields: return only specific fields
        :return: list of OntologyNode objects
        '''
        @query_getter('ontologyNode', ['id', 'originalId', 'termShortName', 'json', 'ontology { id }'])
        def _get_ontology_node(obj, filter=None, fields=None):
            pass
        return [OntologyNode(**dict({'compendium': self.compendium}, **o)) for o in _get_ontology_node(self.compendium, filter=filter, fields=fields)]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


    @staticmethod
    def using(compendium):
        cls = get_compendium_object(OntologyNode)
        return cls(compendium=compendium)

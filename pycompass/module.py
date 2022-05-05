import pandas as pd

'''
A Module class is an interface for creation of modules starting from bio_features and sample_sets. Names can be genes and conditions (conditions can be sample_sets or samples in case of sample_sets with 
only one sample).
if nothing is specified pick up random genes and conditions. In only one of the two is specified infer the other, as a parameter accepts the number of the missing features (either
genes or conditions).
Clustering: default biclustering
Meta information: if nothing is specified provides only gene names and condition names otherwise accepts a series of annotation terms (experiment, platform, tissue, ... ).
There should be a way to visualize all the possible annotation terms to use.
Annotation terms are either quality or quantity and for every new one add a column multiindex. Uniform unit of measurement (accept parameters). Allow to retrieve all meta informations.

Contrasts: should have a name (inferred autamatically during normalization). Should have a short description (inferred by annotation) about what changes (ref vs test) based on the normalization

Clustering: exploit DataFrame or use gene-expression byclustering by default

Module __repr__ should be the dataframe head(5)

Plotting is static: heatmap, network, coexpression-rank, boxplot, histograms, compositional (everything is a PNG so I can use different libraries). Figure size is a parameter. For fancy stuff users
should go with their plotting library of choice.

Modify Module: automatically or manually add genes or conditions (as usual). Allow operations between Modules (return a new one):
a + b sum genes and conditions of a module and b module
a - b subtract genes and conditions of a module and b module
a | b add only conditions of a to b
a ^ b remove only conditions of b from a
a >> b add only genes of a to b
a << b remove only genes from b to a 

View module differences: create another module that combine two summed modules ( + | or >> ) with extra multiindex (named module_1 / module_2 or user defined names)
Plot module differences:

Module selection: use ipywidget to have selection slider on genes and conditions

Module enrichment:


 
'''
import pandas as pd


def plot(*plotting_classes):
    def decorator(module_class):
        def wrapper(*args, **kwargs):
            module = module_class(*args, **kwargs)
            for plot_class in plotting_classes:
                setattr(
                    module_class,
                    str(plot_class.__name__).lower(),
                    plot_class(module)
                )
            return module
        return wrapper
    return decorator


class Plot:
    def __init__(self, module):
        self.module = module

    def plot(self):
        raise NotImplementedError()


class Heatmap(Plot):

    def plot(self):
        pass

    def interactive(self):
        return 'interactive'

    def get_selection(self):
        return 'selection'


@plot(Heatmap)
class Module:

    def __init__(self, *args, **kwargs):
        self._compendium = kwargs.get('compendium', None)
        self.df = kwargs.get('df', None)

    def __iter__(self):
        for k, v in self.df.to_dict().items():
            yield k, v

    def __str__(self):
        if self.df is not None:
            return self.df.head(5).__str__()

    def __repr__(self):
        if self.df is not None:
            return self.df.head(5).__repr__()

    def __add__(self, other):
        pass

    def __rshift__(self, other):
        bf = list(set(self.df.index).union(set(other._df.index)))
        ss = list(self.df.columns)
        return self._compendium.module(biofeatures=bf, samplesets=ss)

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
from ipywidgets import interact
import ipywidgets as widgets
import numpy as np
import matplotlib
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt


def plot(*plotting_classes):
    def decorator(module_class):
        def wrapper(*args, **kwargs):
            module = module_class(*args, **kwargs)
            for plot_class in plotting_classes:
                setattr(
                    module,
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


class Network(Plot):

    def __init__(self, module):
        super().__init__(module)
        self.nx_graph = nx.Graph()

    def plot(self, interactive=False, corr=0.7, *args, **kwargs):
        if interactive:
            return self.__interactive__(corr=corr, *args, **kwargs)
        plot_f = self.__plot__(*args, **kwargs)
        return plot_f(corr=corr)

    def __plot__(self, *args, **kwargs):
        def _plot(corr=0.7):
            self.nx_graph = nx.Graph()

            _corr = self.module.df.T.corr()
            _idx = _corr.index.get_level_values(self.module.df.index.names[-1])
            for i, i_name in enumerate(_idx):
                for j, j_name in enumerate(_idx[i + 1:]):
                    self.nx_graph.add_edge(i_name, j_name, weight=_corr.iloc[i, i + 1 + j])
            for i, attr in enumerate(self.module.df.index.names):
                _attr = {}
                for node in self.module.df.index:
                    _attr.update({node[-1]: node[i]})
                nx.set_node_attributes(self.nx_graph, _attr, name=attr)

            fig, ax = plt.subplots(figsize=(15, 10))

            label = kwargs.get('label', self.module.df.index.names[-1])
            labels = nx.get_node_attributes(self.nx_graph, label)

            layout = kwargs.get('layout', 'circular')
            layout_fun = getattr(nx, layout + '_layout')
            pos = layout_fun(self.nx_graph)
            widths = nx.get_edge_attributes(self.nx_graph, 'weight')
            nx.draw_networkx(self.nx_graph, pos, width=[0 for _ in widths.values()], ax=ax, with_labels=True, labels=labels)
            return nx.draw_networkx_edges(self.nx_graph, pos, ax=ax,
                                          width=[2 if np.abs(c) >= corr else 0 for c in widths.values()],
                                          edge_color=['black' if c >= 0 else 'red' for c in widths.values()])

        return _plot

    def __interactive__(self, *args, **kwargs):
        plot_f = self.__plot__(*args, **kwargs)
        x = widgets.FloatSlider(
            value=kwargs.get('corr', 0.7),
            min=0,
            max=1,
            step=0.1,
            description='PCC',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            # layout=Layout(width='500px')
        )

        interact(plot_f,
                 corr=x
                 )


class Heatmap(Plot):

    def __init__(self, module):
        super().__init__(module)
        self._x = np.array([0, self.module.df.shape[0]])
        self._y = np.array([0, self.module.df.shape[1]])

    def plot(self,
             interactive=False,
             force_squares=True,
             cluster_lines=True,
             *args, **kwargs):
        if interactive:
            return self.__interactive__(force_squares, cluster_lines=cluster_lines, *args, **kwargs)
        plot_f = self.__plot__(force_squares, cluster_lines=cluster_lines, *args, **kwargs)
        return plot_f(
            [0, self.module.df.shape[0]],
            [0, self.module.df.shape[1]]
        )

    def get_selection(self):
        df = self.module.df.iloc[self._x[0]:self._x[1], self._y[0]:self._y[1]]
        return Module(compendium=self.module._compendium, df=df, name='Selection ' + self.module.name)


    def __squared_cluster_map(self, df, cell_size, *args, **kwargs):
        # Thanks to Diziet Asahi
        # https://stackoverflow.com/questions/52806878/seaborn-clustermap-fixed-cell-size
        # Calulate the figure size, this gets us close, but not quite to the right place
        dpi = matplotlib.rcParams['figure.dpi']
        marginWidth = matplotlib.rcParams['figure.subplot.right'] - matplotlib.rcParams['figure.subplot.left']
        marginHeight = matplotlib.rcParams['figure.subplot.top'] - matplotlib.rcParams['figure.subplot.bottom']
        Ny, Nx = df.shape
        figWidth = (Nx * cell_size / dpi) / 0.8 / marginWidth
        figHeigh = (Ny * cell_size / dpi) / 0.8 / marginHeight
        kwargs['figsize'] = (figWidth, figHeigh)
        # do the actual plot
        grid = sns.clustermap(df,
                              *args,
                              **kwargs
                              )

        # calculate the size of the heatmap axes
        axWidth = (Nx * cell_size) / (figWidth * dpi)
        axHeight = (Ny * cell_size) / (figHeigh * dpi)

        # resize heatmap
        ax_heatmap_orig_pos = grid.ax_heatmap.get_position()
        grid.ax_heatmap.set_position([ax_heatmap_orig_pos.x0, ax_heatmap_orig_pos.y0,
                                      axWidth, axHeight])

        # resize dendrograms to match
        ax_row_orig_pos = grid.ax_row_dendrogram.get_position()
        grid.ax_row_dendrogram.set_position([ax_row_orig_pos.x0, ax_row_orig_pos.y0,
                                             ax_row_orig_pos.width, axHeight])
        ax_col_orig_pos = grid.ax_col_dendrogram.get_position()
        grid.ax_col_dendrogram.set_position([ax_col_orig_pos.x0, ax_heatmap_orig_pos.y0 + axHeight,
                                             axWidth, ax_col_orig_pos.height])
        return grid

    def __annotate_columns__(self, col_annotation_terms):
        unique = set()
        lut = {}
        _d = {}
        for _term, _lut in col_annotation_terms.items():
            if type(_lut) == dict:
                lut.update(_lut)
            _d[_term] = pd.Series(self.module.df.columns.get_level_values(_term))
            unique.update(set(_d[_term].unique()))
        unique = list(unique)
        if not lut:
            palette = sns.color_palette("Set2", len(unique))
            lut = dict(zip(unique, palette))

        col_colors = pd.DataFrame([
            pd.Series(self.module.df.columns.get_level_values(term), index=self.module.df.columns).map(lut)
            for term in col_annotation_terms]).T
        return col_colors, pd.DataFrame(_d), lut, unique

    def __annotate_rows__(self, row_annotation_terms):
        unique = set()
        lut = {}
        _d = {}
        for _term, _lut in row_annotation_terms.items():
            if type(_lut) == dict:
                lut.update(_lut)
            _d[_term] = pd.Series(self.module.df.index.get_level_values(_term))
            unique.update(set(_d[_term].unique()))
        unique = list(unique)
        if not lut:
            palette = sns.color_palette("Set2", len(unique))
            lut = dict(zip(unique, palette))

        row_colors = pd.DataFrame([
            pd.Series(self.module.df.index.get_level_values(term), index=self.module.df.index).map(lut)
            for term in row_annotation_terms]).T
        return row_colors, pd.DataFrame(_d), lut, unique

    def __plot__(self, force_squares, title=False, cell_size=25, *args, **kwargs):
        def _plot(x, y):
            self._x = x
            self._y = y
            # new default
            max_h = 15
            max_w = 18
            kwargs['cmap'] = kwargs.get('cmap', sns.diverging_palette(140, 10, as_cmap=True, center="dark"))
            kwargs['figsize'] = kwargs.get('figsize', (max_w, max_h))
            kwargs['yticklabels'] = kwargs.get(
                'yticklabels',
                self.module.df.index.get_level_values('biofeatures').tolist()
            )
            kwargs['xticklabels'] = kwargs.get(
                'xticklabels',
                self.module.df.columns.get_level_values('samplesets').tolist()
            )

            # annotation colors
            col_annotation_terms = {}
            row_annotation_terms = {}
            to_remove = []
            col_colors = None
            row_colors = None
            row_lines = []
            col_lines = []
            for k, v in kwargs.items():
                if k == 'col_colors' or k == 'row_colors':
                    continue
                if k.endswith('_lines'):
                    line_term = k.replace('_lines', '')
                    to_remove.append(k)
                    if line_term in self.module.df.columns.names and v:
                        col_lines.append(line_term)
                    if line_term in self.module.df.index.names and v:
                        row_lines.append(line_term)
                if k.endswith('_colors'):
                    annotation_term = k.replace('_colors', '')
                    to_remove.append(k)
                    if annotation_term in self.module.df.columns.names:
                        col_annotation_terms[annotation_term] = v
                    if annotation_term in self.module.df.index.names:
                        row_annotation_terms[annotation_term] = v
            for arg_to_remove in to_remove:
                del kwargs[arg_to_remove]
            if col_annotation_terms:
                col_colors, col_hm, col_lut, col_unique = self.__annotate_columns__(col_annotation_terms)
            if row_annotation_terms:
                row_colors, row_hm, row_lut, row_unique = self.__annotate_rows__(row_annotation_terms)
            kwargs['col_colors'] = kwargs.get('col_colors', col_colors)
            kwargs['row_colors'] = kwargs.get('row_colors', row_colors)

            if force_squares:
                g = self.__squared_cluster_map(
                    self.module.df.iloc[x[0]:x[1], y[0]:y[1]],
                    cell_size=cell_size,
                    row_cluster=False,
                    col_cluster=False,
                    *args, **kwargs)
            else:
                g = sns.clustermap(
                    self.module.df.iloc[x[0]:x[1], y[0]:y[1]],
                    row_cluster=False,
                    col_cluster=False,
                    *args,
                    **kwargs)

            # lines
            for row_line in row_lines:
                cls = [str(i) for i in self.module.df.index.get_level_values(row_line)]
                prev_pos = 0
                prev_value = cls[prev_pos]
                for i, x in enumerate(cls):
                    if x != prev_value:
                        g.ax_heatmap.axhline(y=i, linewidth=2, color="w")
                    prev_value = x
            for col_line in col_lines:
                cls = [str(i) for i in self.module.df.columns.get_level_values(col_line)]
                prev_pos = 0
                prev_value = cls[prev_pos]
                for i, x in enumerate(cls):
                    if x != prev_value:
                        g.ax_heatmap.axvline(x=i, linewidth=2, color="w")
                    prev_value = x

            # title
            if title is None:
                g.fig.suptitle(None)
            elif title:
                g.fig.suptitle(title)
            else:
                g.fig.suptitle(self.module.name)
            g.fig.subplots_adjust(hspace=0.1)
            g.ax_cbar.set_position((1.05, 0.5, .04, .4))
            g.ax_cbar.set_aspect(5)
            g.ax_heatmap.yaxis.set_label_position('left')
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
            g.ax_heatmap.tick_params(left=True, labelleft=True, right=False, labelright=False)
            # col_colors
            if col_colors is not None:
                for level in col_annotation_terms:
                    for label in col_hm[level].unique():
                        _a = g.ax_col_dendrogram.bar(0, 0, color=col_lut[label], label=label, linewidth=0)
                g.ax_col_dendrogram.legend(title=' '.join(col_annotation_terms), loc="center", ncol=len(col_annotation_terms))
            # row_colors
            if row_colors is not None:
                for level in row_annotation_terms:
                    for label in row_hm[level].unique():
                        _a = g.ax_row_dendrogram.bar(0, 0, color=row_lut[label], label=label, linewidth=0)
                g.ax_row_dendrogram.legend(title=' '.join(row_annotation_terms), loc="center",
                                           ncol=len(row_annotation_terms))

            return g

        return _plot

    def __interactive__(self, force_squares, *args, **kwargs):
        plot_f = self.__plot__(force_squares, *args, **kwargs)
        x = widgets.IntRangeSlider(
            value=[0, self.module.df.shape[0]],
            min=0,
            max=self.module.df.shape[0],
            step=1,
            description='biofeatures',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            # layout=Layout(width='500px')
        )
        y = widgets.IntRangeSlider(
            value=[0, self.module.df.shape[1]],
            min=0,
            max=self.module.df.shape[1],
            step=1,
            description='samplesets',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            # layout=Layout(width='500px')
        )
        interact(plot_f,
                 x=x,
                 y=y
                 )


class Annotate(object):
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        if 'biofeatures' in kwargs:
            self.__annotate_biofeatures__(kwargs['biofeatures'])
        if 'samplesets' in kwargs:
            self.__annotate_samplesets__(kwargs['samplesets'])

    def __annotate_biofeatures__(self, biofeatures_terms):
        for term in biofeatures_terms:
            if term in ['name', 'description']:
                bf_idx = self.module.df.index.get_level_values('biofeatures').tolist()
                bfs = [getattr(bf, term) for bf in self.module._compendium.query('biofeatures').filter(id_In=bf_idx).fields(term)]
                self.module.df[term] = pd.Series(bfs, index=self.module.df.index)
                self.module.df = self.module.df.reset_index().set_index([term] + list(self.module.df.index.names))
            else:
                pass # sparql

    def __annotate_samplesets__(self, samplesets_terms):
        for term in samplesets_terms:
            if term in ['name', 'shortAnnotationDescription']:
                ss_idx = self.module.df.columns.get_level_values('samplesets').tolist()
                ss = [getattr(ss, term) for ss in self.module._compendium.query('samplesets').filter(id_In=ss_idx).fields(term)]
                df = self.module.df.T
                df[term] = pd.Series(ss, index=df.index)
                df = df.reset_index().set_index([term] + list(df.index.names))
                self.module.df = df.T
            else:
                pass  # sparql


@plot(Heatmap, Network)
class Module:

    def __init__(self, *args, **kwargs):
        self._compendium = kwargs.get('compendium', None)
        self.df = kwargs.get('df', None)
        self.name = kwargs.get('name', 'Module')
        self.annotate = Annotate(self)

    def extends(self, target='biofeatures', first=50, score=0.9, *args, **kwargs):
        connections = []
        if self._compendium.version:
            connections.append('version:"{}"'.format(self._compendium.version))
        if self._compendium.database:
            connections.append('database:"{}"'.format(self._compendium.database))
        if self._compendium.normalization:
            connections.append('normalization:"{}"'.format(self._compendium.normalization))
        qs = self._compendium.query('scoreRankMethods')
        id_type = 'samplesetIds'
        if target == 'biofeatures':
            query = '''
            {{
              scoreRankMethods(compendium:"{compendium}", {connections}) {{
                    biologicalFeatures
              }}
            }}
            '''.format(
                compendium=self._compendium.name,
                connections=', ' + ','.join(connections)
            )
            json = qs.__run_query__(query)
            methods = json['data']['scoreRankMethods']['biologicalFeatures']
            method = kwargs.get('method', methods[0])
            if method not in methods:
                raise Exception('Invalid method argument. Allowed values are ' + ' '.join(methods))
            ids = ','.join('"{}"'.format(x) for x in self.df.columns.get_level_values('samplesets'))
        else:
            query = '''
            {{
              scoreRankMethods(compendium:"{compendium}", {connections}) {{
                    sampleSets
              }}
            }}
            '''.format(
                compendium=self._compendium.name,
                connections=', ' + ','.join(connections)
            )
            json = qs.__run_query__(query)
            methods = json['data']['scoreRankMethods']['sampleSets']
            method = kwargs.get('method', methods[0])
            if method not in methods:
                raise Exception('Invalid method argument. Allowed values are ' + ' '.join(methods))
            id_type = 'biofeaturesIds'
            ids = ','.join('"{}"'.format(x) for x in self.df.index.get_level_values('biofeatures'))

        qs = self._compendium.query('ranking')
        query = '''
        {{
            ranking(compendium: "{compendium}", rankTarget: "{target}", rank: "{method}",
                {id_type}: [{ids}], {connections}) {{
                id,
                name,
                type,
                value
            }}
        }}'''.format(
                compendium=self._compendium.name,
                connections=', ' + ','.join(connections),
                target=target,
                method=method,
                id_type=id_type,
                ids=ids
        )
        json = qs.__run_query__(query)
        data = json['data']['ranking']
        _values = np.array(data['value'])
        limit = min(first, len(_values[_values >= score]))
        _data = set()
        bf_idx = self.df.index.get_level_values('biofeatures').tolist()
        ss_idx = self.df.columns.get_level_values('samplesets').tolist()
        for _d in data['id']:
            if len(_data) == limit:
                break
            if target == 'biofeatures':
                if _d not in bf_idx:
                    _data.add(_d)
            elif target == 'samplesets':
                if _d not in ss_idx:
                    _data.add(_d)
        if target == 'biofeatures':
            bf_idx += list(_data)
        elif target == 'samplesets':
            ss_idx += list(_data)
        return self._compendium.module(
            name='Extended {target} {name}'.format(target=target, name=self.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )

    def __iter__(self):
        for k, v in self.df.to_dict().items():
            yield k, v

    def __str__(self):
        if self.df is not None:
            return self.df.head(5).__str__()

    def __repr__(self):
        if self.df is not None:
            return self.df.head(5).__repr__()

    def __sub__(self, other):
        '''
        a - b subtract genes and conditions of a module and b module
        '''
        bf_idx = list(set(self.df.index.get_level_values('biofeatures')) - set(other.df.index.get_level_values('biofeatures')))
        ss_idx = list(set(self.df.columns.get_level_values('samplesets')) - set(other.df.columns.get_level_values('samplesets')))
        if not len(bf_idx):
            raise Exception('Resulting biofeatures list is empty!')
        if not len(ss_idx):
            raise Exception('Resulting samplesets list is empty!')
        new_module = self._compendium.module(
            name='{module_1} - {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )
        return new_module

    def __lshift__(self, other):
        '''
        a << b remove only genes from b to a
        '''
        bf_idx = list(
            set(self.df.index.get_level_values('biofeatures')) - set(other.df.index.get_level_values('biofeatures')))
        ss_idx = list(set(self.df.columns.get_level_values('samplesets')))
        if not len(bf_idx):
            raise Exception('Resulting biofeatures list is empty!')
        new_module = self._compendium.module(
            name='{module_1} << {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )
        return new_module

    def __xor__(self, other):
        '''
        a ^ b remove only conditions of b from a
        '''
        bf_idx = list(set(self.df.index.get_level_values('biofeatures')))
        ss_idx = list(
            set(self.df.columns.get_level_values('samplesets')) - set(other.df.columns.get_level_values('samplesets')))
        if not len(ss_idx):
            raise Exception('Resulting samplesets list is empty!')
        new_module = self._compendium.module(
            name='{module_1} ^ {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )

        return new_module

    def __or__(self, other):
        '''
        a | b add only conditions of a to b
        '''
        self_label = 'module_left'
        other_label = 'module_right'
        bf_idx = list(set(self.df.index.get_level_values('biofeatures')))
        ss_idx = list(set(self.df.columns.get_level_values('samplesets')).union(
            set(other.df.columns.get_level_values('samplesets'))))
        new_module = self._compendium.module(
            name='{module_1} | {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )
        # add new multiindex biofeatures
        return self.__modify_multiindex__(other, new_module, self_label, other_label)

    def __add__(self, other):
        '''
        a + b sum genes and conditions of a module and b module
        '''
        self_label = 'module_left'
        other_label = 'module_right'
        bf_idx = list(set(self.df.index.get_level_values('biofeatures')).union(
            set(other.df.index.get_level_values('biofeatures'))))
        ss_idx = list(set(self.df.columns.get_level_values('samplesets')).union(
            set(other.df.columns.get_level_values('samplesets'))))
        new_module = self._compendium.module(
            name='{module_1} + {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )
        # add new multiindex biofeatures
        return self.__modify_multiindex__(other, new_module, self_label, other_label)

    def __rshift__(self, other):
        '''
        a >> b add only genes of a to b
        '''
        self_label = 'module_left'
        other_label = 'module_right'
        bf_idx = list(set(self.df.index.get_level_values('biofeatures')).union(
            set(other.df.index.get_level_values('biofeatures'))))
        ss_idx = list(set(self.df.columns.get_level_values('samplesets')))
        new_module = self._compendium.module(
            name='{module_1} >> {module_2}'.format(module_1=self.name, module_2=other.name),
            biofeatures=bf_idx,
            samplesets=ss_idx
        )
        # add new multiindex biofeatures
        return self.__modify_multiindex__(other, new_module, self_label, other_label)

    def __modify_multiindex__(self, other, new_module, self_label, other_label):
        first_bf_list = self.df.index.get_level_values('biofeatures').tolist()
        second_bf_list = other.df.index.get_level_values('biofeatures').tolist()
        first_name = self.name
        second_name = other.name
        _first = [first_name if i[-1] in first_bf_list else None for i in new_module.df.index.tolist()]
        _second = [second_name if i[-1] in second_bf_list else None for i in new_module.df.index.tolist()]
        new_module.df[self_label] = pd.Series(_first, index=new_module.df.index)
        new_module.df[other_label] = pd.Series(_second, index=new_module.df.index)
        names = [self_label, other_label] + list(new_module.df.index.names)
        new_module.df = new_module.df.reset_index().set_index(names)

        first_ss_list = self.df.columns.get_level_values('samplesets').tolist()
        second_ss_list = other.df.columns.get_level_values('samplesets').tolist()
        first_name = self.name
        second_name = other.name
        new_module.df = new_module.df.T
        _first = [first_name if i[-1] in first_ss_list else None for i in new_module.df.index.tolist()]
        _second = [second_name if i[-1] in second_ss_list else None for i in new_module.df.index.tolist()]
        new_module.df[self_label] = pd.Series(_first, index=new_module.df.index)
        new_module.df[other_label] = pd.Series(_second, index=new_module.df.index)
        names = [self_label, other_label] + list(new_module.df.index.names)
        new_module.df = new_module.df.reset_index().set_index(names)
        new_module.df = new_module.df.T

        # missing index and columns
        missing_index = [x for x in self.df.index.names if x not in new_module.df.index.names]
        missing_columns = [x for x in self.df.columns.names if x not in new_module.df.columns.names]

        new_module.annotate(biofeatures=missing_index, samplesets=missing_columns)

        return new_module
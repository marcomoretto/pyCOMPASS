from pycompass.query import run_query


class Plot:

    def get_plot(self, type=None, format='html', **kwargs):
        '''
        Plot the module

        :param type: the plot type
        :param format: html or json
        :return: the plot in the specified format
        '''

        base = 'plotDistribution'
        base_query = '''\
                {{\
                    {base}(compendium:"{compendium}",
                    plotType:"{plot_type}" 
                    {normalization} {filter}) {{\
                        {output}\
                    }}\
                }}\
                '''
        if type == 'sample_sets_magnitude_distribution':
            base = 'plotDistribution'
            query = base_query.format(
                base=base,
                compendium=self._compendium.compendium_name,
                plot_type=type,
                normalization=', normalization:"' + self._normalization + '"',
                filter=', biofeaturesIds:[' + ','.join(['"' + x + '"' for x in self._biological_features]) + ']',
                output=format
            )
        elif type == 'sample_sets_coexpression_distribution':
            base = 'plotDistribution'
            query = base_query.format(
                base=base,
                compendium=self._compendium.compendium_name,
                plot_type=type,
                normalization=', normalization:"' + self._normalization + '"',
                filter=', biofeaturesIds:[' + ','.join(['"' + x + '"' for x in self._biological_features]) + ']',
                output=format
            )
        elif type == 'biological_features_standard_deviation_distribution':
            base = 'plotDistribution'
            query = base_query.format(
                base=base,
                compendium=self._compendium.compendium_name,
                plot_type=type,
                normalization=', normalization:"' + self._normalization + '"',
                filter=', samplesetIds:[' + ','.join(['"' + x + '"' for x in self._sample_sets]) + ']',
                output=format
            )
        elif type == 'module_heatmap_expression':
            base = 'plotHeatmap'
            sort_by = ', sortBy:"{}"'.format(kwargs['sort_by']) if 'sort_by' in kwargs else ''
            alternative_coloring_value = 'true' if bool(kwargs.get('alternative_coloring', False)) else 'false'
            alternative_coloring = ', alternativeColoring:{}'.format(alternative_coloring_value)
            query = base_query.format(
                base=base,
                compendium=self._compendium.compendium_name,
                plot_type=type,
                normalization='',
                filter=', biofeaturesIds:[' + ','.join(['"' + x + '"' for x in self._biological_features]) + '] '
                       ', samplesetIds:[' + ','.join(['"' + x + '"' for x in self._sample_sets]) + '] ' +
                       sort_by + alternative_coloring,
                output=format
            )

        return run_query(self._compendium.connection.url, query)['data'][base][format]



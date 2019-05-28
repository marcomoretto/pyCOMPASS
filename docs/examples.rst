Examples
========

Here you can find a collection of random examples.

Connect
-------

.. code-block:: python

  '''
  Connect to a COMPASS GraphQL endpoint and retrieve
  all available compendium as a list of Compendium objects
  '''
  from pycompass import Connect, Compendium, Module, Sample, Platform, Experiment, Ontology, SampleSet, Module

  url = 'http://compass.fmach.it/graphql'
  conn = Connect(url)
  compendia = conn.get_compendia()

Get data sources
----------------

.. code-block:: python

  '''
  Get all (and only) the data sources name
  (public or private databases) from one Compendium
  '''
  ds = compendia[0].get_data_sources(fields=['sourceName'])

Get Samples
-----------

.. code-block:: python

  '''
  Get the first 10 samples from one Compendium
  as a list of Sample objects
  '''
  s = Sample.using(compendia[0]).get(filter={'first': 10})

Get Platforms
-------------

.. code-block:: python

  '''
  Get the first 2 platforms from one Compendium
  as a list of Platform objects with only the
  platformAccessId field
  '''
  plts = Platform.using(compendia[0]).get(fields=['platformAccessId'], filter={'first': 2})

  '''
  Get the platform related to a sample
  '''
  sp = s[0].platform

Get Experiments
---------------

.. code-block:: python

  '''
  Get the first experiment from one Compendium
  as a list of Experiment objects
  '''
  es = Experiment.using(compendia[0]).get(filter={'first': 1})

  '''
  Get the experiment related to a sample
  '''
  se = s[0].experiment

Get Samples
-----------

.. code-block:: python

  '''
  Get all samples measuread with a given Platform
  '''
  s = Sample.using(compendia[0]).by(platform=plts[0])

Get Ontology
------------

.. code-block:: python

  '''
  Get ontologies as list of Ontology objects given a name
  and retrieve the structure of one as JSON
  '''
  os = Ontology.using(compendia[0]).get(filter={'name': 'Gene ontology'})
  st = os[0].structure

Get SampleSet
-------------

.. code-block:: python

  '''
  Get the first 2 sample sets as list of SampleSet objects
  or by a given Sample object
  '''
  ss = SampleSet.using(compendia[0]).get(filter={'first': 2})
  ss = SampleSet.using(compendia[0]).by(samples=s[:1])

Get BiologicalFeature
---------------------

.. code-block:: python

  '''
  Get biological feature as a list of BiologicalFeature objects
  given a list of names
  '''
  bf = BiologicalFeature.using(compendia[0]).get(filter={'name_In': ['VIT_00s0332g00160', 'VIT_00s0396g00010', 'VIT_00s0505g00030']})

Create Module
-------------
.. code-block:: python

  '''
  Create a Module object given a list of SampleSet objects
  the BiologicalFeature objects are inferred
  '''
  mod1 = Module.using(compendia[0]).create(samplesets=ss)

  '''
  Create a Module object given a list of BiologicalFeature objecst
  the SampleSet objects are inferred
  '''
  mod2 = Module.using(compendia[0]).create(biofeatures=bf)

  '''
  Create a Module as union of 2 other modules
  '''
  mod3 = Module.union(mod1, mod2)

  '''
  Create a Module as intersection of 2 other modules
  '''
  mod4 = Module.intersection(mod3, mod2)

  '''
  Create a Module as difference of 2 other modules
  '''
  mod5 = Module.difference(mod1, mod2)

Plot
----

.. code-block:: python

  '''
  Get the module heatmap plot as HTML file to show on a browser
  '''
  html = Plot(mod1).plot_heatmap(alternativeColoring=True)

  '''
  Get the module coexpression network plot as HTML file to show on a browser
  '''
  html = Plot(mod1).plot_network()

  '''
  Get different biological feature or sample sets distribution plots
  based on module's values as HTML file to show on a browser
  '''
  html = Plot(mod1).plot_distribution(plot_type='sample_sets_magnitude_distribution')

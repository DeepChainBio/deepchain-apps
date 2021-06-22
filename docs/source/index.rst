.. deepchain-apps documentation master file, created by
   sphinx-quickstart on Fri Jun 18 15:00:02 2021.
   You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

=======================
What is deepchain-apps?
=======================

.. tip::  **deepchain-apps** package is a component of the `deepchain.bio <https://deepchain.bio/>`_ software. Please take 5 minutes to create an account or take a look at the deepchain-apps public `hub <https://app.deepchain.bio/hub/apps>`_

DeepChain apps is a collaborative framework that allows the user to create scorers to evaluate protein sequences. The apps can be deployed on the deepchain plateform. These scorers can be either classifier, predictor, or whatever functions that take a 
protein sequence in input and return a scalar.

The `deepchain-apps <https://pypi.org/project/deepchain-apps/>`_ package is hosting in pypi. The `deep-chain-apps <https://github.com/DeepChainBio/deep-chain-apps>`_ template to build an app can be found on github.

DeepChainBio repository
------------------------
The Github `DeepChainBio <https://github.com/DeepChainBio>`_  is hosting 3 python packages to help to build apps.

   * `bio-transformers <https://github.com/DeepChainBio/bio-transformers>`_: This package is a wrapper on top of the state of the art protein design models.
   * `bio-datasets <https://github.com/DeepChainBio/bio-datasets>`_ allows downloading protein sequences datasets and pre-computed embeddings.
   * `deep-chain-apps <https://github.com/DeepChainBio/deep-chain-apps>`_ provide an Apps template.

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Getting Started

   getting_started/install.rst
   getting_started/quick_start.rst

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Documentation

   documentation/deepchainapps.rst
   documentation/cli.rst
   documentation/deepchain
   documentation/logging

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Tutorial

   tutorial/apps_building.rst
   tutorial/train_model
   

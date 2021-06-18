Installation
============

Bio-transformers can be installed in Python 3.7 and external python dependencies are mainly defined in `requirements`_.
There are multiple different methods to install Bio-transformers:

1. Clone `deepchain-apps`_ and create a virtual environment using `Anaconda`_ / `Miniconda`_ (**recommended**).
2. Install directly from PyPI release without cloning `deepchain-apps`_.


Install via Conda
-----------------
The recommended method is to install Bio-transformers in a dedicated virtual
environment using `Anaconda`_ / `Miniconda`_.


.. code:: bash

    conda create --name bio-transformers python=3.7 -y
    conda activate deepchain-apps
    pip install deepchain-apps

.. _Quick Start: quick_start.html
.. _Anaconda: https://docs.anaconda.com/anaconda/install
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _deepchain-apps: https://pypi.org/project/deepchain-apps/
.. _requirements: https://github.com/DeepChainBio/bio-transformers/blob/main/requirements.txt

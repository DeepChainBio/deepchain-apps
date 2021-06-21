App structure
-------------

When creating an app with the CLI, you will download the current github folder with the following structure.

.. important:: The main class must be named `App` in app.py ins the src folder.

.. code-block:: text

    ├── README.md # explains how to create an app
    ├── __init__.py # __init__ file to create python module
    ├── checkpoint
    │   ├── __init__.py
    │   └── Optionnal : model.pt # optional: model to be used in app must be placed there
    ├── examples
    │   ├── app_with_checkpoint.py # example: app example with checkpoint
    │   └── torch_classifier.py # example: show how to train a neural network with pre-trained embeddings
    └── src
        ├── DESC.md # Desciption file of the application, feel free to put a maximum of information.
        ├── __init__.py
        ├── app.py # main application script. Main class must be named App.
        └── Optional : model.py # file to register the models you use in app.py.
        └── tags.json # file to register the tags on the hub.

Tags
----
For your app to be visible and well documented, tags should be filled to precise at least the tasks section. It will be really useful to retrieve it from deepchain hub.

* tasks
* librairies
* embeddings
* datasets
* device

.. important:: If you want your app to benefit from deepchain' GPU, set device to "gpu" in tags. It will run on "cpu" by default.

Special method in apps
----------------------
Every apps inherit from the ``DeepChainApp`` class that provide some special method. This method are useful for loading extra files
in your app. 

* Use method ``get_checkpoint_path(__file__)`` to get the absolute path of the file in the ``checkpoint`` folder.
* Use method ``get_filepath(__file__,file)`` to get the absolute path of the file in the ``src`` folder.

Where to find apps?
-----------------

All the applications that have been released publicly can be found on the `hub <https://app.deepchain.bio/hub/apps>`_

DeepchainHub 
^^^^^^^^^^^^

.. image::  images/deepchainhub.jpeg
    :alt: deepchainhub

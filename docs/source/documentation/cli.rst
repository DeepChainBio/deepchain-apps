===
CLI
===

Commands description
--------------------

The CLI provides 5 main commands:

* **login**: you need to supply the token provide on the platform (PAT: personal access token).

.. code-block:: bash

    deepchain login

* **create**: create a folder with a template app file.

.. code-block:: bash

    deepchain login

* **deploy**: the code and checkpoint are deployed on the platform, you can select your app in the interface on the platform.

    * with checkpoint upload
    
    .. code-block:: bash

        deepchain deploy my_application --checkpoint
    
    * only the code
    
    .. code-block:: bash

        deepchain deploy my_application

* **apps**: command related to apps actions.

    * Get info on all local/upload apps
    
    .. code-block:: bash

        deepchain apps --infos
    
    .. important::  The apps show with this command are not synchronized with deepchain-hub.
    
    * Remove all local apps (files & config):
    
    .. code-block:: bash

        deepchain apps --reset
    
    * Remove a specific application (files & config):
    
    .. code-block:: bash

        deepchain apps --delete my_application
    
    * List all public apps:
    
    .. code-block:: bash

        deepchain apps --public

* **download** : download locally an app deployed on deepchain hub

.. code-block:: bash

    deepchain download user.name@mail.com/AppName AppName
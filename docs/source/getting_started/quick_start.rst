How it works
============

If you want to create and deploy an app on deepchain hub, you could use the command provided in the deepchain-apps package. 
Below are the main commands that should be used in a terminal:

Basic cli
---------

1. Put your personal access token (PAT) to deploy the application on `deepchain.bio <https://deepchain.bio/>`_

.. code-block:: bash

    deepchain login

2. Create an application based on an empty template.

.. code-block:: bash

    deepchain create myapplication

3. Or download a pre-existing app. First list available apps.

.. code-block:: bash

    deepchain apps --public

   `----------------------------------------------------------------
    APP                                        USERNAME             
    ----------------------------------------------------------------
    FullInfluenzaBinding                 k.eloff@hotmail.co.za      
    OntologyPredict                    stj.grimbly@instadeep.com    
    DiseaseRiskApp                     jb.sevestre@instadeep.com    
  


Download app command  :  deepchain download username/app app


Apps description at : https://app.deepchain.bio/hub/apps`

Then download it.

.. code-block:: bash

    deepchain download user.name/appname appname
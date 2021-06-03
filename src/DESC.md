# Description
This app is a base application that compute the loglikelihood (the probability that a protein is natural or not).

We use a bio-transformers with a 'protbert'  model as backend to make the computation of logits

# Tags
Fill the tags.json file in this folder:

- tasks: str -> kind of task perform (classifier, regressor, indicator...)
- libraries: str -> libraries used for the app
- embeddings: str -> embedding model if used.
- datasets: str -> dataset name if used.
- device: ["cpu"]: str -> By default, the app are launched on cpu on deepchain, put device to "gpu" to benefit GPU
                          and accelerate the optimization process during score computation.


## libraries
- pytorch>=1.5.0

## tasks
- probability
- transformers
- unsupervised

## embeddings
- ESM

## datasets
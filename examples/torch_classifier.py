"""
Module that provide a classifier template to train a model on embeddings.
With use the pathogen vs human dataset as an example. The embedding of 100k proteins come 
from the protBert model.
The model is built with pytorch_ligthning, a wrapper on top of 
pytorch (similar to keras with tensorflow)
Feel feel to build you own model if you want to build a more complex one
"""

from biodatasets import list_datasets, load_dataset
from deepchain.models import MLP
from deepchain.models.utils import (
    confusion_matrix_plot,
    dataloader_from_numpy,
    model_evaluation_accuracy,
)
from sklearn.model_selection import train_test_split

# Load embedding and target dataset
pathogen = load_dataset("pathogen")
_, y = pathogen.to_npy_arrays(input_names=["sequence"], target_names=["class"])
embeddings = pathogen.get_embeddings("sequence", "protbert", "cls")

X_train, X_val, y_train, y_val = train_test_split(embeddings, y[0], test_size=0.3)

train_dataloader = dataloader_from_numpy(X_train, y_train, batch_size=32)
test_dataloader = dataloader_from_numpy(X_val, y_val, batch_size=32)

# Build a multi-layer-perceptron on top of embedding

# The fit method can handle all the arguments available in the
# 'trainer' class of pytorch lightening :
#               https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
# Example arguments:
# * specifies all GPUs regardless of its availability :
#               Trainer(gpus=-1, auto_select_gpus=False, max_epochs=20)

mlp = MLP(input_shape=X_train.shape[1])
mlp.fit(train_dataloader, epochs=5)
mlp.save_model("model.pt")

# Model evaluation
prediction, truth = model_evaluation_accuracy(test_dataloader, mlp)

# Plot confusion matrix
confusion_matrix_plot(truth, (prediction > 0.5).astype(int), ["0", "1"])

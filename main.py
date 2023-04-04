"""
Playground for getting started with using PyTorch for building machine learning applications.

Author: Zico da Silva
"""
import lib
import matplotlib.pyplot as plt
import torch

# Obtain the data.
n_dim = 28 * 28
n_output = 10
use_multi_label = True
training_set, testing_set = lib.load_mnist_dataset(use_multi_label=use_multi_label)
# Define a model.
n_features = 32
model = torch.nn.Sequential(
    torch.nn.Linear(n_dim, n_features),
    torch.nn.ReLU(),
    torch.nn.Linear(n_features, n_output),
)
# Setup a basic SGD optimisation.
learning_rate = 0.1
# opt = lib.BasicOptimiser(model.parameters(), learning_rate)
opt = torch.optim.SGD(model.parameters(), learning_rate)
# Train model.
results = lib.train_model(model, opt, training_set, testing_set, 150)
# Predict a random image.
print(lib.classify(model, "/Users/zico/.fastai/data/mnist_png/testing/8/290.png", n_dim, use_multi_label=use_multi_label))
# Post-processing of results.
plot_results = False
if plot_results:
    fig = plt.figure()
    plt.plot(results["train_loss"])
    plt.plot(results["test_loss"])
    plt.plot(results["test_accuracy"])
    plt.show()


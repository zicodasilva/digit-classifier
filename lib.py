"""
Library of functions created for easy use with PyTorch and Fast.ai.

Author: Zico da Silva
"""
import os
import functools
from typing import Tuple, Dict, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from fastai.vision.all import untar_data, URLs, Image, Path

def load_mnist_dataset(input_img_size: int = 28, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    # Get data.
    print("Download dataset (if not downloaded already)...")
    path = untar_data(URLs.MNIST)
    # Load images into a train and testing tensor(s) and normalise to range 0 - 1.0.
    img_trans = transforms.ToTensor()
    X, X_val = [], []
    y, y_val = [], []
    for digit in range(10):
        img_train_data = torch.stack([img_trans(Image.open(o)).squeeze() for o in Path(os.path.join(path, "training", str(digit))).ls()])
        img_test_data = torch.stack([img_trans(Image.open(o)).squeeze() for o in Path(os.path.join(path, "testing", str(digit))).ls()])
        X.append(img_train_data)
        y.append([digit] * img_train_data.size(0))
        X_val.append(img_test_data)
        y_val.append([digit] * img_test_data.size(0))
    # Create training set - make sure to flatten the image data into a 1D vector.
    X = torch.cat(X).view(-1, input_img_size * input_img_size)
    # Create labels for the output, i.e. binary classification, with 1 representing a 3 and 0 representing a 7.
    y = torch.tensor(functools.reduce(lambda a, b: a+b, y)).unsqueeze(1)
    # Put together into a dataset that PyTorch supports.
    data_set = list(zip(X, y))
    # Do the same for the validation set.
    X_val = torch.cat(X_val).view(-1, input_img_size * input_img_size)
    y_val = torch.tensor(functools.reduce(lambda a, b: a+b, y_val)).unsqueeze(1)
    validation_set = list(zip(X_val, y_val))
    # Create DataLoader object so that we can perform mini-batch training.
    training_set = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    return training_set, testing_set

# Step 1: Create loss function.
def mnist_loss(predictions: torch.Tensor, targets: torch.Tensor):
    output = predictions.log_softmax(dim=1)
    return (-output[range(targets.size(0)), targets.squeeze()]).mean()
    # return F.cross_entropy(predictions, targets.squeeze()) # Could use this function.
    # return F.nll_loss(output, targets.squeeze()) # Or use this in conjunction with the `output` above.

# Step 2: Create optimiser to perform training loop.
class BasicOptimiser(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None

# Step 3: Calculate gradient i.e. how does the loss change with respect to the weights.
def calc_grad(xb: torch.Tensor, yb: torch.Tensor, model: torch.nn.Module):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
    return loss / xb.size(0)

def train_epoch(model: torch.nn.Module, optimiser: torch.optim.Optimizer, dataset: DataLoader):
    avg_loss = 0.0
    for x, y in dataset:
        avg_loss += calc_grad(x, y, model)
        optimiser.step()
        optimiser.zero_grad()

    return avg_loss.item() / len(dataset)

def calc_batch_accuracy(mdl: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> Tuple[float, float]:
    _, max_idx_class = mdl(X).softmax(dim=1).max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    acc = (max_idx_class == Y.squeeze()).sum().item() / X.size(0)
    batch_loss = 0.0
    with torch.no_grad():
        batch_loss = mnist_loss(mdl(X), Y).item()
    return batch_loss, acc

def validate_epoch(model: torch.nn.Module, dataset: DataLoader) -> Tuple[float, float]:
    loss, acc = [], []
    for xb, yb in dataset:
        l, a = calc_batch_accuracy(model, xb, yb)
        loss.append(l)
        acc.append(a)
    validation_acc = np.stack(acc)
    validation_loss = np.stack(loss)
    return validation_loss.mean(), validation_acc.mean()

def train_model(model: torch.nn.Module, optimiser: torch.optim.Optimizer, training_set: DataLoader, testing_set: DataLoader, num_epochs: int, print_rate: int = 5) -> Dict:
    ret = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }
    for i in range(num_epochs):
        loss = train_epoch(model, optimiser, training_set)
        test_loss, accuracy = validate_epoch(model, testing_set)
        if i % print_rate == 0 or i == num_epochs - 1:
            print(f"epoch: {i}, training loss: {loss:.4f}, test loss: {test_loss:.4f}, test accuracy (%): {accuracy:.3f}")
        ret["train_loss"].append(loss)
        ret["test_loss"].append(test_loss)
        ret["test_accuracy"].append(accuracy)

    return ret

def classify(model: torch.nn.Module, img_path: Union[str, Path], n_dim: int) -> float:
    img_data = transforms.ToTensor()(Image.open(img_path))
    digit = torch.argmax(torch.nn.Softmax(dim=1)(model(img_data.view(-1, n_dim))), dim=1)

    return digit.item()

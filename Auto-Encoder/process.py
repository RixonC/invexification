import numpy as np
import torch
from torchvision.utils import make_grid


def process_data_decay():
    data_decay = "1e-8"
    epochs = 10 * np.arange(100)
    paths = ["epoch-{}.pt".format(i) for i in epochs]
    orig_loss = np.array([torch.load(path)["orig_loss"] for path in paths])
    surr_loss = np.array([torch.load(path)["surr_loss"] for path in paths])
    surr_grad = np.array([torch.load(path)["surr_grad_norm"] for path in paths])
    test_loss = np.array([torch.load(path)["test_loss"] for path in paths])
    pl_ratios = 2 * (float(data_decay) ** 2) * surr_loss / (surr_grad ** 2)
    data = {
        "epochs" : epochs,
        "orig_loss" : orig_loss,
        "surr_loss" : surr_loss,
        "surr_grad" : surr_grad,
        "test_loss" : test_loss,
        "pl_ratios" : pl_ratios,
    }
    torch.save(data, "processed/data-decay-" + data_decay + ".pt")
    torch.save(
        torch.load("epoch-1000.pt"),
        "processed/data-decay-" + data_decay + "-epoch-1000.pt"
    )


def process_weight_decay():
    weight_decay = "1e-8"
    epochs = 10 * np.arange(100)
    paths = ["epoch-{}.pt".format(i) for i in epochs]
    orig_loss = np.array([torch.load(path)["orig_loss"] for path in paths])
    surr_loss = np.array([torch.load(path)["surr_loss"] for path in paths])
    test_loss = np.array([torch.load(path)["test_loss"] for path in paths])
    data = {
        "epochs" : epochs,
        "orig_loss" : orig_loss,
        "surr_loss" : surr_loss,
        "test_loss" : test_loss,
    }
    torch.save(data, "processed/weight-decay-" + weight_decay + ".pt")
    torch.save(
        torch.load("epoch-1000.pt"),
        "processed/weight-decay-" + weight_decay + "-epoch-1000.pt"
    )


def process_no_decay():
    epochs = 10 * np.arange(100)
    paths = ["epoch-{}.pt".format(i) for i in epochs]
    orig_loss = np.array([torch.load(path)["orig_loss"] for path in paths])
    surr_loss = np.array([torch.load(path)["surr_loss"] for path in paths])
    test_loss = np.array([torch.load(path)["test_loss"] for path in paths])
    data = {
        "epochs" : epochs,
        "orig_loss" : orig_loss,
        "surr_loss" : surr_loss,
        "test_loss" : test_loss,
    }
    torch.save(data, "processed/no-decay.pt")
    torch.save(
        torch.load("epoch-1000.pt"),
        "processed/no-decay-epoch-1000.pt"
    )
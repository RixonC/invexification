import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


matplotlib.rcParams["font.family"] = ["Times New Roman"]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["text.usetex"] = True

indices = torch.tensor([125, 137, 145, 149, 156, 165, 180, 191])
fig, ax0 = plt.subplots(figsize=(5, 0.9 * len(indices)))

data = torch.load("processed/no-decay-epoch-1000.pt")
images = data["sample_reconstructed"][indices]
data = torch.load("processed/data-decay-1e-2-epoch-1000.pt")
images = torch.cat([images, data["sample_reconstructed"][indices]], dim=0)
data = torch.load("processed/data-decay-1e-4-epoch-1000.pt")
images = torch.cat([images, data["sample_reconstructed"][indices]], dim=0)
data = torch.load("processed/weight-decay-1e-2-epoch-1000.pt")
images = torch.cat([images, data["sample_reconstructed"][indices]], dim=0)
data = torch.load("processed/weight-decay-1e-4-epoch-1000.pt")
images = torch.cat([images, data["sample_reconstructed"][indices]], dim=0)


permutation = torch.arange(0, 5 * len(indices), step=len(indices))
permutation = torch.cat([permutation + i for i in range(len(indices))])
images = images[permutation]
images = make_grid(images, nrow=5, padding=4)
images = images.transpose(0, 1).transpose(1, 2)
ax0.imshow(images, cmap="gray", interpolation="none")
ax0.set_xticks(132 * np.arange(5) + 66)
ax0.set_yticks([])
ax0.set_xticklabels([
    "No \n Regularization",
    "Ours, \n " + r"$\lambda=0.01$",
    "Ours, \n " + r"$\lambda=0.0001$",
    r"$\ell_2$," + "\n" + r"$\lambda=0.01$",
    r"$\ell_2$," + "\n" + r"$\lambda=0.0001$",
])
ax0.set_yticklabels([])
ax0.xaxis.set_ticks_position('top')
plt.subplots_adjust(top=0.94, bottom=0, left=0, right=1)
plt.savefig("Figure-6.pdf")
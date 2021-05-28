import matplotlib
import matplotlib.pyplot as plt
import torch


values = [1e-1, 1e-2, 1e-3, 1e-4]
labels = [r"$\lambda=0.1$", r"$\lambda=0.01$", r"$\lambda=0.001$", r"$\lambda=0.0001$"]
colors = ["k", "r", "b", "g"]
styles = ["-", "--", ":", "-."]

plt.style.use(["science"])
matplotlib.rcParams["font.family"] = ["Times New Roman"]
matplotlib.rcParams["font.size"] = 10

fig0, axs = plt.subplots(nrows=3, ncols=2, sharey="row", figsize=(5.4, 4))
(ax0, ax3), (ax1, ax4), (ax2, ax5) = axs

ax0.set_title("Our Method")
ax3.set_title(r"$\ell_2$ Regularization")

ax0.set_xlabel("Epoch")
ax1.set_xlabel("Epoch")
ax2.set_xlabel(r"$\lambda$")
ax3.set_xlabel("Epoch")
ax4.set_xlabel("Epoch")

ax2.set_xlim(1e-16, 1)

ax0.set_ylabel("Non \n Regularized \n Loss")
ax1.set_ylabel("Regularized \n Loss")
ax2.set_ylabel("Maximum \n PL Ratio")

ax0.set_ylim(1e-16, 1e2)
ax1.set_ylim(1e-28, 1e2)

for value, label, c, ls in zip(values, labels, colors, styles):
    path = "data-decays/{:.0e}.pt".format(value)
    data = torch.load(path)
    ax0.loglog(data["indices"] + 1, data["orig_loss"], label=label, c=c, ls=ls)
    ax1.loglog(data["indices"] + 1, data["surr_loss"], label=label, c=c, ls=ls)

    path = "weight-decays/{:.0e}.pt".format(value)
    data = torch.load(path)
    ax3.loglog(data["indices"] + 1, data["orig_loss"], label=label, c=c, ls=ls)
    ax4.loglog(data["indices"] + 1, data["surr_loss"], label=label, c=c, ls=ls)

pl_ratios = []
values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

for value in values:
    path = "data-decays/{:.0e}.pt".format(value)
    data = torch.load(path)
    pl = max(2 * (value ** 2) * data["surr_loss"] / (data["surr_grad"] ** 2))
    pl_ratios.append(pl)

ax2.loglog(values, pl_ratios)
ax2.invert_xaxis()

ax4.legend(
    ncol=2,
    columnspacing=1.0,
    handlelength=1.25,
    loc="center",
    bbox_to_anchor=(0.5, -1.25),
)
ax5.axis("off")

ax0.set_yticks([1, 1e-8, 1e-16])
ax1.set_yticks([1, 1e-14, 1e-28])
ax2.set_yticks([1, 1e-14, 1e-28])

plt.subplots_adjust(left=0.17, right=0.99, wspace=0.1, hspace=0.7)
plt.savefig("plot.pdf")

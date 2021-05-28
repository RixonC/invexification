import matplotlib
import matplotlib.pyplot as plt
import torch


plt.style.use(["science"])
matplotlib.rcParams["font.family"] = ["Times New Roman"]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["text.usetex"] = True

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(5, 4), sharey="row")
(ax0, ax4), (ax1, ax5), (ax2, ax6), (ax3, ax7) = axs
colors = ["k", "r", "b", "g"]
styles = ["-", "--", ":", "-."]
ax0.set_title("Our Method")
ax4.set_title(r"$\ell_2$ Regularization")
ax3.set_xlabel("Epoch")
ax6.set_xlabel("Epoch")
ax0.set_ylabel("Non \n Regularized \n Loss")
ax1.set_ylabel("Regularized \n Loss")
ax2.set_ylabel("Test \n Dataset \n Loss")
ax3.set_ylabel("PL Ratio")

ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])

ax0.set_ylim(5e-1, 1e1)
ax1.set_ylim(1e-1, 1e1)
ax2.set_ylim(1e-1, 2e0)
ax3.set_ylim(1e-16, 1e-2)

scale = 1e-6

data = torch.load("processed/no-decay.pt")
label = r"$\lambda=0$"
color = colors[0]
ls = styles[0]
ax0.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax1.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax2.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)
ax4.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax5.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax6.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)

data = torch.load("processed/data-decay-1e-4.pt")
label = r"$\lambda=0.0001$"
color = colors[1]
ls = styles[1]
ax0.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax1.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax2.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)
ax3.semilogy(data["epochs"], data["pl_ratios"], label=label, color=color, ls=ls)

data = torch.load("processed/data-decay-1e-2.pt")
label = r"$\lambda=0.01$"
color = colors[2]
ls = styles[2]
ax0.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax1.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax2.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)
ax3.semilogy(data["epochs"], data["pl_ratios"], label=label, color=color, ls=ls)

data = torch.load("processed/data-decay-1.pt")
label = r"$\lambda=1$"
color = colors[3]
ls = styles[3]
ax0.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax1.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax2.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)
ax3.semilogy(data["epochs"], data["pl_ratios"], label=label, color=color, ls=ls)

data = torch.load("processed/weight-decay-1e-4.pt")
label = r"$\lambda=0.0001$"
color = colors[1]
ls = styles[1]
ax4.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax5.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax6.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)

data = torch.load("processed/weight-decay-1e-2.pt")
label = r"$\lambda=0.01$"
color = colors[2]
ls = styles[2]
ax4.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax5.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax6.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)

data = torch.load("processed/weight-decay-1.pt")
label = r"$\lambda=1$"
color = colors[3]
ls = styles[3]
ax4.semilogy(data["epochs"], scale * data["orig_loss"], label=label, color=color, ls=ls)
ax5.semilogy(data["epochs"], scale * data["surr_loss"], label=label, color=color, ls=ls)
ax6.semilogy(data["epochs"], scale * data["test_loss"], label=label, color=color, ls=ls)

ax6.legend(ncol=2, columnspacing=1.0, handlelength=1.25, loc="center", bbox_to_anchor=(0.5, -1))
ax7.axis("off")
plt.subplots_adjust(wspace=0.075)
plt.savefig("Figure-3.pdf")
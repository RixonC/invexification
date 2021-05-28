import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

plt.style.use(["science"])
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10
# matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(5, 2.5))
gs = GridSpec(nrows=2, ncols=4)

ax0 = fig.add_subplot(gs[0, 0])
g = lambda v: 4 * v.sin() + v.pow(2) / 2 + 8

x = torch.linspace(-5, 5, steps=100)
ax0.plot(x, g(x), c="k")
global_min = ax0.scatter([-1.2524], [4.98530], c="g")
local_max = ax0.scatter([2.1333], [13.6592], c="r")
local_min = ax0.scatter([3.5953], [12.7099], c="b")


# --------------------------------- lambda = 5 ---------------------------------


x = torch.linspace(-6, 6, steps=100)
ax1 = fig.add_subplot(gs[0:2, 1])
valley = ax1.plot(x, -g(x) / 5, c="g")[0]
ax1.set_ylim(-4, 2)

x = torch.linspace(-6, 6, steps=12)
p = torch.linspace(-4, 2, steps=12)
X, P = torch.meshgrid(x, p)
X.requires_grad = True
P.requires_grad = True
Y = g(X) + 5 * P
Y = Y.pow(2).sum()
Y.backward()
ax1.quiver(X.detach().numpy(), P.detach().numpy(), -X.grad.numpy(), -P.grad.numpy())

xt = torch.tensor([6.0], requires_grad=True)
pt = torch.tensor([0.0], requires_grad=True)
gd = torch.optim.SGD([xt, pt], lr=0.015)
x_path = []
p_path = []
for _ in range(100000):
    x_path.append(xt.item())
    p_path.append(pt.item())
    gd.zero_grad()
    y = (g(xt) + 5 * pt).pow(2).sum()
    y.backward()
    gd.step()
gd_path = ax1.plot(x_path, p_path, lw=2, c="r")[0]
print("({:.2f},{:.2f})".format(x_path[-1], p_path[-1]))


# --------------------------------- lambda = 2 ---------------------------------


x = torch.linspace(-6, 6, steps=100)
ax2 = fig.add_subplot(gs[0:2, 2])
ax2.plot(x, -g(x) / 2, c="g")[0]
ax2.set_ylim(-8, 2)

x = torch.linspace(-6, 6, steps=12)
p = torch.linspace(-8, 2, steps=12)
X, P = torch.meshgrid(x, p)
X.requires_grad = True
P.requires_grad = True
Y = g(X) + 2 * P
Y = Y.pow(2).sum()
Y.backward()
ax2.quiver(X.detach().numpy(), P.detach().numpy(), -X.grad.numpy(), -P.grad.numpy())

xt = torch.tensor([6.0], requires_grad=True)
pt = torch.tensor([0.0], requires_grad=True)
gd = torch.optim.SGD([xt, pt], lr=0.015)
x_path = []
p_path = []
for _ in range(100000):
    x_path.append(xt.item())
    p_path.append(pt.item())
    gd.zero_grad()
    y = (g(xt) + 2 * pt).pow(2).sum()
    y.backward()
    gd.step()
ax2.plot(x_path, p_path, lw=2, c="r")[0]
print("({:.2f},{:.2f})".format(x_path[-1], p_path[-1]))


# -------------------------------- lambda = 0.1 --------------------------------


x = torch.linspace(-6, 6, steps=100)
ax3 = fig.add_subplot(gs[0:2, 3])
ax3.plot(x, -g(x) / 0.1, c="g")
ax3.set_ylim(-120, 6)

x = torch.linspace(-6, 6, steps=8)
p = torch.linspace(-120, 10, steps=24)
X, P = torch.meshgrid(x, p)
X.requires_grad = True
P.requires_grad = True
Y = g(X) + 0.1 * P
Y = Y.pow(2).sum()
Y.backward()
ax3.quiver(X.detach().numpy(), P.detach().numpy(), -X.grad.numpy(), -P.grad.numpy())

xt = torch.tensor([6.0], requires_grad=True)
pt = torch.tensor([0.0], requires_grad=True)
gd = torch.optim.SGD([xt, pt], lr=0.015)
x_path = []
p_path = []
for _ in range(100000):
    x_path.append(xt.item())
    p_path.append(pt.item())
    gd.zero_grad()
    y = (g(xt) + 0.1 * pt).pow(2).sum()
    y.backward()
    gd.step()
ax3.plot(x_path, p_path, lw=2, c="r")
print("({:.2f},{:.2f})".format(x_path[-1], p_path[-1]))


# ------------------------------------------------------------------------------


ax0.set_title("Objective Function", size=10)
ax1.set_title(r"$\lambda=5$", size=10)
ax2.set_title(r"$\lambda=2$", size=10)
ax3.set_title(r"$\lambda=0.1$", size=10)

ax0.set_xlabel(r"$x$")
ax1.set_xlabel(r"$x$")
ax2.set_xlabel(r"$x$")
ax3.set_xlabel(r"$x$")

ax0.set_ylabel(r"$f(x)$")
ax1.set_ylabel(r"$p$")
ax2.set_ylabel(r"$p$")
ax3.set_ylabel(r"$p$")

fig.legend(
    handles=(global_min, local_max, local_min, valley, gd_path),
    labels=(
        "Global Minimum",
        "Local Maximum",
        "Local Minimum",
        "Global Minima",
        "GD Path",
    ),
    loc="lower left",
    borderaxespad=0.25,
)
plt.subplots_adjust(wspace=1.0, left=0.09, right=0.99, bottom=0.15)
plt.savefig("plot.pdf")

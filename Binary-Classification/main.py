import argparse
import torch
from numpy import log10
from tqdm import tqdm

torch.manual_seed(0)
dtype = torch.float64

parser = argparse.ArgumentParser()
parser.add_argument("--data_decay", type=float, default=0.0)
parser.add_argument("--weight_decay", type=float, default=0.0)
args = parser.parse_args()

assert args.data_decay == 0.0 or args.weight_decay == 0.0
data_decay = torch.tensor(args.data_decay).to(dtype=dtype)
weight_decay = torch.tensor(args.weight_decay).to(dtype=dtype)
weight_decay_sq = weight_decay * weight_decay
learn_rate = torch.tensor(2.0).to(dtype=dtype)
num_epochs = "10,000,000"
num_epochs = int(num_epochs.replace(",", ""))

z = torch.randn(8, dtype=dtype)
A = torch.randn(64, 8, dtype=dtype)
A *= torch.logspace(
    start=0, end=-8, steps=A.size(1), base=2.0, dtype=dtype
)
b = A.mv(z).sigmoid()

At = A.t()
x = torch.zeros(A.size(1), dtype=dtype)
p = torch.zeros(A.size(0), dtype=dtype)
hf = torch.tensor(0.5).to(dtype=dtype)
tmp1 = torch.zeros_like(b)
tmp2 = torch.zeros_like(b)

def fun(X, P):
    tmp1.zero_().addmv_(A, X).sigmoid_()  # sigmoid(A.mv(X))
    tmp2.zero_().add_(tmp1).sub_(b)  # sigmoid(A.mv(X)) - b
    tmp1.addcmul_(tmp1, tmp1, value=-1)  # sigmoid'(A.mv(X))
    orig = torch.dot(tmp2, tmp2)
    orig.mul_(hf)  # 0.5 * || sigmoid(A.mv(X)) - b ||^2
    tmp2.add_(P, alpha=data_decay)  # sigmoid(A.mv(X)) - b + data_decay * P
    loss = torch.dot(tmp2, tmp2)
    loss.add_(X @ X, alpha=weight_decay_sq)  # 0.5 * weight_decay^2 * || X ||^2
    loss.mul_(hf)  # 0.5 * || sigmoid(A.mv(X)) - b + data_decay * P ||^2
    tmp1.mul_(tmp2)  # sigmoid'(A.mv(X))*(sigmoid(A.mv(X)) - b + data_decay * P)
    grad_x = torch.addmv(X, At, tmp1, beta=weight_decay_sq)  # d/dx f(x,p)
    # grad_x = At.mv(tmp1)  # d/dx f(x,p)
    tmp2.mul_(data_decay)  # data_decay * (sigmoid(A.mv(X)) - b + data_decay * P)
    grad_p = tmp2  # d/dp f(x,p)
    return orig, loss, grad_x, grad_p

assert fun(z, 0.0)[0].item() == 0.0
indices = torch.logspace(0, int(log10(num_epochs)), 100, dtype=torch.long)
indices = torch.unique(indices, sorted=True) - 1
orig_loss = torch.zeros_like(indices, dtype=dtype)
surr_loss = torch.zeros_like(indices, dtype=dtype)
surr_grad = torch.zeros_like(indices, dtype=dtype)
j = 0
for i in tqdm(range(num_epochs), miniters=num_epochs // 1000, maxinterval=1e4):
    f1, f2, gx, gp = fun(x, p)
    x.sub_(gx, alpha=learn_rate)
    p.sub_(gp, alpha=learn_rate)
    if i == indices[j]:
        orig_loss[j].add_(f1)
        surr_loss[j].add_(f2)
        surr_grad[j].add_(torch.dot(gx, gx)).add_(torch.dot(gp, gp)).sqrt_()
        j += 1

results = {
    "indices": indices,
    "orig_loss": orig_loss,
    "surr_loss": surr_loss,
    "surr_grad": surr_grad,
}
if data_decay.item() != 0.0:
    path = "data-decays/{:.0e}.pt".format(data_decay.item())
else:
    path = "weight-decays/{:.0e}.pt".format(weight_decay.item())
torch.save(results, path)
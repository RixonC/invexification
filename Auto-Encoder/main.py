import argparse
import torch
from model import DataDecayWrapper, VAE
from sklearn.datasets import fetch_lfw_people
from torch.nn.utils import parameters_to_vector as ptv
from torch_optimizer import Adahessian as Optimizer
from torchvision.utils import save_image
from tqdm import trange


@torch.no_grad()
def evaluate(
    device, dtype, epoch, model, optimizer, X_train_split, X_test, weight_decay
):
    model.eval()
    criterion = torch.nn.MSELoss(reduction="sum")
    m = len(X_train_split)
    orig_loss = 0.0
    surr_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for j in range(m):
        inputs = X_train_split[j]
        outputs = model.model(inputs)
        loss = criterion(outputs, inputs)
        orig_loss += loss.item()
        with torch.enable_grad():
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss = loss + (weight_decay ** 2) * ptv(model.parameters()).pow(2).sum()
        loss.backward()
        model.step()
        surr_loss += loss.item()
    surr_grad = sum([p.grad.data.pow(2).sum() for p in model.parameters()])
    surr_grad = surr_grad.sqrt().item()
    test_loss = criterion(model.model(X_test), X_test).item()
    sample_num = 256
    sample_latent = torch.randn(sample_num, model.model.latent_dim)
    sample_latent = sample_latent.to(device, dtype)
    sample_reconstructed = model.model(X_test[:sample_num]).cpu()
    results = {
        "orig_loss": orig_loss,
        "surr_loss": surr_loss,
        "surr_grad_norm": surr_grad,
        "test_loss": test_loss,
        "sample_reconstructed": sample_reconstructed,
    }
    torch.save(results, "epoch-{}.pt".format(epoch))
    save_image(sample_reconstructed, "img-{}.png".format(epoch))
    model.train()


def get_lfw(device, dtype):
    X, _ = fetch_lfw_people(
        resize=1.0,
        return_X_y=True,
        slice_=(slice(68, 196), slice(64, 192)),
    )
    X = torch.from_numpy(X).to(dtype=torch.float32).view(13233, 128, 128, -1)
    X.transpose_(2, 3).transpose_(1, 2)
    X -= X.min()
    X /= X.max()
    return X[:10000].to(device, dtype), X[10000:].to(device, dtype)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    dtype = torch.float32
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=1e-0)
    parser.add_argument("--data-decay", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    assert args.data_decay == 0.0 or args.weight_decay == 0.0

    batch_size = 10
    X_train, X_test = get_lfw(device, dtype)
    X_train_split = torch.split(X_train, batch_size)
    model = VAE(img_size=X_train.size(2))
    model = DataDecayWrapper(model, batch_size, args.data_decay, X_train.size())
    model = model.to(device=device, dtype=dtype)
    criterion_avg = torch.nn.MSELoss()
    optimizer = Optimizer(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay ** 2
    )
    create_graph = optimizer.__class__.__name__ == "Adahessian"

    evaluate(
        device,
        dtype,
        0,
        model,
        optimizer,
        X_train_split,
        X_test,
        args.weight_decay,
    )
    with trange(1000) as t1:
        for i in t1:
            for j in range(len(X_train_split)):
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    reconstruction = model(X_train_split[j])
                    loss = criterion_avg(X_train_split[j], reconstruction)
                loss.backward(create_graph=create_graph)
                optimizer.step()
                model.step()
            t1.set_postfix(loss="{:.2e}".format(loss.item()))
            if (i + 1) % 10 == 0:
                evaluate(
                    device,
                    dtype,
                    i + 1,
                    model,
                    optimizer,
                    X_train_split,
                    X_test,
                    args.weight_decay,
                )


if __name__ == "__main__":
    main()

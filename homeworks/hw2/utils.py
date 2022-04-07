import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ProgressBar
from scipy.optimize import bisect
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from tqdm.auto import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class LossTracker(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"]
        val_loss = trainer.callback_metrics["val_loss"]
        self.train_loss.append(train_loss.cpu().item())
        self.val_loss.append(val_loss.cpu().item())


class PlotCallback(Callback):
    def __init__(self, plot_frequency=5, epochs_to_plot=None):
        super().__init__()
        if epochs_to_plot:
            plot_frequency = 1e3
        else:
            epochs_to_plot = []

        self.plot_frequency = plot_frequency
        self.epochs_to_plot = epochs_to_plot

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if epoch % self.plot_frequency == 0 or epoch in self.epochs_to_plot:
            pl_module.plot(f"Epoch {epoch}")
            plt.show()


class NotebookProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)

        return bar


class ImageFlow(pl.LightningModule):
    def __init__(
            self,
            flows,
            base_dist,
            lr=1e-3,
    ):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)

        if base_dist == "uniform":
            self.base_dist = Uniform(torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
        elif base_dist == "normal":
            self.base_dist = Normal(torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
        elif base_dist == "beta":
            self.base_dist = Beta(torch.FloatTensor([5.0]), torch.FloatTensor([5.0]))
        else:
            raise NotImplementedError

        self.lr = lr

    def forward(self, x):
        # The forward function is only used for visualizing the graph
        return self.log_prob(x.float())

    def flow(self, x):
        z, log_det = x.float(), torch.zeros_like(x)
        for flow_module in self.flows:
            z, log_det = flow_module(z, log_det)
        return z, log_det

    def invert(self, z):
        for flow_module in reversed(self.flows):
            z = flow_module.invert(z)
        return z

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return self.base_dist.log_prob(z).sum(dim=1) + log_det.sum(dim=1)

    # Compute loss as negative log-likelihood
    def loss(self, x):
        return -self.log_prob(x).mean()

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


def train_model(flows, train_loader, val_loader, train_args):
    module = ImageFlow(
        flows,
        train_args["base_dist"],
        train_args["lr"],
    )
    loss_tracker = LossTracker()
    progress_bar = NotebookProgressBar()
    callbacks = [loss_tracker, progress_bar]

    trainer = pl.Trainer(
        max_epochs=train_args["epochs"],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        checkpoint_callback=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return module, loss_tracker.train_loss, loss_tracker.val_loss


class MixtureCDFFlow(nn.Module):
    def __init__(self, n_components=4):
        super().__init__()
        self.loc = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_scale = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.mixture_dist = Normal
        self.n_components = n_components

    def forward(self, x, log_det=0):
        return self.flow(x, log_det)

    def flow(self, x, log_det=0):
        # set up mixture distribution
        weights = (
            F.softmax(self.weight_logits, dim=0).unsqueeze(0).repeat(x.shape[0], 1)
        )

        mixture_dist = self.mixture_dist(self.loc, self.log_scale.exp())
        x_repeat = x.unsqueeze(1).repeat(1, self.n_components)

        # z = cdf of x
        z = torch.clamp((mixture_dist.cdf(x_repeat) * weights).sum(dim=1), 0, 1 - 1e-5)

        # log_det = log dz/dx = log pdf(x)
        log_det += (mixture_dist.log_prob(x_repeat).exp() * weights).sum(dim=1).log()

        return z, log_det

    def invert(self, z):
        # Find the exact x via bisection such that f(x) = z
        results = []
        for z_elem in z:
            def f(x):
                return self.flow(torch.tensor(x).unsqueeze(0))[0] - z_elem

            x = bisect(f, -20, 20)
            results.append(x)
        return torch.tensor(results).reshape(z.shape)


class MLP(nn.Module):
    def __init__(self, size_list):
        super().__init__()
        layers = []
        for input_size, output_size in zip(size_list[:-1], size_list[1:]):
            layers.extend(
                [
                    nn.Linear(in_features=input_size, out_features=output_size),
                    nn.ReLU(),
                ]
            )
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConditionalMixtureCDFFlow(nn.Module):
    def __init__(self, n_layers=3, hidden_size=64, cond_dim=1, n_components=4):
        super().__init__()
        self.mlp = MLP([cond_dim] + n_layers * [hidden_size] + [n_components * 3])
        self.mixture_dist = Normal
        self.n_components = n_components

    def forward(self, x, cond, log_det=0):
        return self.flow(x, cond, log_det)

    def flow(self, x, cond, log_det):
        # set up mixture distribution
        loc, log_scale, weight_logits = torch.chunk(self.mlp(cond), 3, dim=1)
        weights = F.softmax(weight_logits, dim=1)

        mixture_dist = self.mixture_dist(loc, log_scale.exp())

        x_repeat = x.repeat(1, self.n_components)

        # z = cdf of x
        z = torch.clamp((mixture_dist.cdf(x_repeat) * weights).sum(dim=1), 0, 1 - 1e-5)

        # log_det = log dz/dx = log pdf(x)
        log_det += (mixture_dist.log_prob(x_repeat).exp() * weights).sum(dim=1).log()

        return z, log_det

    def invert(self, z, cond):
        # Find the exact x via bisection such that f(x) = z
        results = []
        for z_elem in z:
            def f(x):
                return self.flow(torch.tensor(x).unsqueeze(0), cond, 0)[0] - z_elem

            x = bisect(f, -20, 20)
            results.append(x)
        return torch.tensor(results).reshape(z.shape)


class AutoregressiveFlow(nn.Module):
    def __init__(
            self,
            dim1_n_components=5,
            dim2_n_components=5,
            mlp_hidden_size=64,
            mlp_n_layers=3,
    ):
        super().__init__()
        self.dim1_flows = MixtureCDFFlow(dim1_n_components)
        self.dim2_flows = ConditionalMixtureCDFFlow(
            mlp_n_layers, mlp_hidden_size, n_components=dim2_n_components
        )

    def forward(self, x, log_det=0):
        return self.flow(x, log_det)

    def flow(self, x, log_det):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, log_det_1 = self.dim1_flows(x1.squeeze(), log_det)
        z2, log_det_2 = self.dim2_flows(x2, x1, log_det)
        z = torch.hstack([z1.unsqueeze(1), z2.unsqueeze(1)])
        log_det = torch.hstack([log_det_1.unsqueeze(1), log_det_2.unsqueeze(1)])

        return z, log_det


class AffineFlow(nn.Module):
    def __init__(
            self,
            side
    ):
        super().__init__()
        if side == "left":
            self.mask = torch.FloatTensor([1, 0])
        elif side == "right":
            self.mask = torch.FloatTensor([0, 1])
        self.side = side
        self.scale = nn.Parameter(torch.randn(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.randn(1), requires_grad=True)
        self.mlp = MLP([2, 64, 64, 64, 2])

    def forward(self, x, log_det):
        return self.flow(x, log_det)

    def flow(self, x, log_det):
        g_scale, g_scale_shift = torch.chunk(self.mlp(x * self.mask), 2, dim=1)
        log_scale = self.scale * F.tanh(g_scale) + self.scale_shift
        log_scale = log_scale * (1 - self.mask)
        g_scale_shift = g_scale_shift * (1 - self.mask)
        z = torch.exp(log_scale) * x + g_scale_shift

        log_det += log_scale

        return z, log_det

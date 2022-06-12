import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.auto import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class MetricsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.lr = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args) -> None:
        train_loss = trainer.callback_metrics["train_loss"]
        self.train_loss.append(train_loss.cpu().item())

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"]
        val_loss = trainer.callback_metrics["val_loss"]
        lr = trainer.callback_metrics["lr"]
        self.train_loss.append(train_loss.cpu().item())
        self.val_loss.append(val_loss.cpu().item())
        self.lr.append(lr.cpu().item())


class NotebookProgressBar(TQDMProgressBar):
    def __int__(self, refresh_rate):
        super(NotebookProgressBar, self).__int__()
        self._refresh_rate = refresh_rate

    def init_validation_tqdm(self):
        bar = tqdm(disable=True)

        return bar


class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)

    # Compute loss as negative log-likelihood
    def nll(self, x):
        logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1)  # batch_size x d
        return F.cross_entropy(logits, x.long())

    def get_distribution(self):
        distribution = F.softmax(self.logits, dim=0)
        return distribution.detach().cpu().numpy()


class ARModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        nll = self.model.nll(batch)

        self.log("lr",
                 self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)

        self.log("train_loss",
                 nll,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return nll

    def validation_step(self, batch, batch_idx):
        nll = self.model.nll(batch)
        self.log("val_loss",
                 nll,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return nll

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


def train_model(model, train_dl, val_dl, train_args):
    module = ARModule(model, lr=train_args['lr'])
    metrics_tracker = MetricsTracker()
    progress_bar = NotebookProgressBar(refresh_rate=50)
    callbacks = [metrics_tracker, progress_bar]

    trainer = pl.Trainer(
        max_epochs=train_args["epochs"],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        gpus=train_args.get("gpu", 1 if torch.cuda.is_available() else 0),
        callbacks=callbacks,
    )

    val_res = trainer.validate(module, dataloaders=val_dl, verbose=False)
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    metrics_tracker.val_loss = [val_res[0]['val_loss']] + metrics_tracker.val_loss

    return module, metrics_tracker


if __name__ == '__main__':
    from deepul.hw1_helper import *

    train_data, test_data = q1_sample_data_1()
    d = 20
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    # model
    model = Histogram(d)

    # train
    train_args = dict(
        epochs=20,
        lr=1e-1,
        gpu=0,
    )
    module, metrics_tracker = train_model(model, train_dl, test_dl, train_args)

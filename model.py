from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR, mean_squared_error as MSE
from torch import nn
from skimage import color


class Model(nn.Module):
    def __init__(self, name: str, q_values: int, h: callable, h_inv: callable, criterion: callable):
        super(Model, self).__init__()
        self.name = name
        self.h = h
        self.h_inv = h_inv
        self.train_loss = []
        self.val_loss = []

        self.criterion = criterion

        self.epochs = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = self.conv_layer(2, 1, 64, stride=2)
        self.conv2 = self.conv_layer(2, 64, 128, stride=2)
        self.conv3 = self.conv_layer(3, 128, 256, stride=2)
        self.conv4 = self.conv_layer(3, 256, 512)
        self.conv5 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv6 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv7 = self.conv_layer(3, 512, 512)
        self.conv8 = self.conv_layer(3, 512, 256, stride=0.5, normalise=False)

        self.distribution = nn.Conv2d(256, q_values, 1)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=1e-3)

    @staticmethod
    def conv_layer(depth: int, in_channels: int, out_channels: int, stride: Union[float, int] = 1, dilation: int = 1,
                   normalise: bool = True) -> nn.Sequential:
        layers = []
        for i in range(depth):
            if stride == 0.5 and i == 0:
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                        out_channels,
                                        3,
                                        padding=dilation,
                                        stride=stride if i == depth - 1 and stride != 0.5 else 1,
                                        dilation=dilation))
            layers.append(nn.ReLU())

        if normalise:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def normalise(x):
        return (x / 50) - 1

    def forward(self, x):
        x = self.normalise(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.distribution(x)
        return x

    @staticmethod
    def cross_entropy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
        return -torch.sum((nn.functional.log_softmax(y_pred, dim=1)) * y) / (y.shape[0] * y.shape[2] * y.shape[3])

    @staticmethod
    def cross_entropy_rebalanced(y_pred: torch.Tensor, y: torch.Tensor, v: callable) -> float:
        weights = v(y)
        return -torch.sum((nn.functional.log_softmax(y_pred, dim=1)) * y * weights) / (y.shape[0] * y.shape[2] * y.shape[3])

    def save(self):
        torch.save({
            "epochs": self.epochs,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }, f"CHECKPOINTS/{self.name}_{self.epochs}.pt")

    def load(self, name: str):
        checkpoint = torch.load(f"CHECKPOINTS/{name}.pt")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.epochs = checkpoint["epochs"]

    def encode(self, image: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        image = image.to(self.device)
        l = image[:, :1, :, :]
        shape = (l.shape[-2] // 4, l.shape[-1] // 4)
        ab = torch.nn.functional.interpolate(image[:, 1:, :, :], size=shape, mode="bilinear")
        z = self.h_inv(ab)
        return l, z

    def decode(self, l: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ab = self.h(z)
        ab = torch.nn.functional.interpolate(ab, size=l.shape[-2:], mode="bilinear")
        image = torch.cat((l, ab), dim=1)
        return image

    @staticmethod
    def lab_to_rgb(image: torch.Tensor) -> np.ndarray:
        return np.clip(color.lab2rgb(image.cpu().permute(0, 2, 3, 1)), 0, 1)

    def plot(self, show=True):
        batches = np.linspace(0, self.epochs, len(self.train_loss))
        plt.plot(batches, self.train_loss, label="Training loss")

        batches = np.linspace(0, self.epochs, len(self.val_loss))
        plt.plot(batches, self.val_loss, label="Validation loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.name}.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def train_model(self, train: torch.utils.data.DataLoader, val: torch.utils.data.DataLoader, epochs: int):
        self.train()

        for _ in range(epochs):
            time = datetime.now()

            self.train()

            epoch_train_loss = 0
            for i, batch in enumerate(train):
                self.optimiser.zero_grad()

                l, z = self.encode(batch)
                z_pred = self(l)
                loss = self.criterion(z_pred, z)

                loss.backward()
                self.optimiser.step()

                epoch_train_loss += loss.item()
                self.train_loss.append(loss.item())
                # print(f"Batch: {i}, Loss: {loss.item()}")
            epoch_train_loss /= len(train)

            epoch_val_loss = 0
            if val:
                with torch.no_grad():
                    self.eval()
                    for i, batch in enumerate(val):
                        l, z = self.encode(batch)
                        z_pred = self(l)
                        loss = self.criterion(z_pred, z)

                        epoch_val_loss += loss.item()
                epoch_val_loss /= len(val)
            self.val_loss.append(epoch_val_loss)

            self.epochs += 1

            print(
                f"Epoch: {self.epochs}, Train loss: {epoch_train_loss}, Validation loss: {epoch_val_loss}, Time: {datetime.now() - time}")

            if self.epochs % 10 == 0:
                self.save()

        self.plot(show=False)

    def test(self, test: torch.utils.data.DataLoader):
        rmse_ab_metric = 0.0
        rmse_metric = 0.0
        psnr_metric = 0.0
        ssim_metric = 0.0

        with torch.no_grad():
            self.eval()
            for i, batch in enumerate(test):
                batch_pred = self.predict(batch)

                batch_rgb = self.lab_to_rgb(batch)
                batch_pred_rgb = self.lab_to_rgb(batch_pred)

                rmse, rmse_ab, psnr, ssim = 0, 0, 0, 0
                for i in range(batch.shape[0]):
                    pred_ab = batch_pred[i, 1:, :, :].cpu().numpy()
                    ab = batch[i, 1:, :, :].cpu().numpy()

                    rmse_ab += np.sqrt(MSE(pred_ab, ab))

                    pred_rgb = batch_pred_rgb[i]
                    rgb = batch_rgb[i]

                    rmse += np.sqrt(MSE(pred_rgb, rgb))
                    psnr += PSNR(rgb, pred_rgb, data_range=1)
                    ssim += SSIM(pred_rgb, rgb, datarange=1, channel_axis=2, multichannel=True)

                rmse_ab_metric += rmse_ab / batch.shape[0]
                rmse_metric += rmse / batch.shape[0]
                psnr_metric += psnr / batch.shape[0]
                ssim_metric += ssim / batch.shape[0]

            rmse_ab_metric /= len(test)
            rmse_metric /= len(test)
            psnr_metric /= len(test)
            ssim_metric /= len(test)

            print(f"RMSE AB: {rmse_ab_metric}")
            print(f"RMSE: {rmse_metric}")
            print(f"PSNR: {psnr_metric}")
            print(f"SSIM: {ssim_metric}")

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        l, z = self.encode(image)
        with torch.no_grad():
            z = self(l)

        return self.decode(l, z)

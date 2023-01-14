from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR, mean_squared_error as MSE
from torch import nn
from skimage import color


class Model(nn.Module):
    """
    Default colorization model
    """
    def __init__(self, name: str, q_values: int, h: callable, h_inv: callable, criterion: callable):
        """
        Create a model
        :param name: Name of the model
        :param q_values: Number of values in the quantisation of the AB channels
        :param h: Function that maps a distribution to AB channels
        :param h_inv: Function that maps AB channels to a distribution
        :param criterion: Loss function
        """
        super(Model, self).__init__()
        self.name = name
        self.h = h
        self.h_inv = h_inv
        self.train_loss = []
        self.val_loss = []

        self.criterion = criterion

        self.epochs = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create all 8 convolutional layers in the model
        self.conv1 = self.conv_layer(2, 1, 64, stride=2)
        self.conv2 = self.conv_layer(2, 64, 128, stride=2)
        self.conv3 = self.conv_layer(3, 128, 256, stride=2)
        self.conv4 = self.conv_layer(3, 256, 512)
        self.conv5 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv6 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv7 = self.conv_layer(3, 512, 512)
        self.conv8 = self.conv_layer(3, 512, 256, stride=0.5, normalise=False)

        # Output a distribution over the q values for every pixel
        self.distribution = nn.Conv2d(256, q_values, 1)

        # Optimiser initialised with parameters from the paper
        self.optimiser = torch.optim.Adam(self.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=1e-3)

    @staticmethod
    def conv_layer(depth: int, in_channels: int, out_channels: int, stride: Union[float, int] = 1, dilation: int = 1,
                   normalise: bool = True) -> nn.Sequential:
        """
        Helper function to create a convolutional "layer" as described in the paper
        Only the last layer is normalised and uses a stride
        :param depth: How many repeated ReLU and Conv2d layers
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride of the convolution, can be 0.5 to upsample
        :param dilation: Dilation of the convolution
        :param normalise: Whether to use batch normalisation
        :return: A sequential layer
        """
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
    def normalise(x: torch.Tensor) -> torch.Tensor:
        """
        Normalise the input to the range [-1, 1]
        :param x: Lightness channel
        :return: Normalised lightness channel
        """
        return (x / 50) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :param x: Lightness channel
        :return: Distribution over all quantised AB values Q
        """
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
        """
        Cross entropy loss over a two-dimensional tensor
        :param y_pred: Predicted distribution
        :param y: Actual distribution
        :return: Cross entropy loss
        """
        return -torch.sum((nn.functional.log_softmax(y_pred, dim=1)) * y) / (y.shape[0] * y.shape[2] * y.shape[3])

    @staticmethod
    def cross_entropy_rebalanced(y_pred: torch.Tensor, y: torch.Tensor, v: callable) -> float:
        """
        Cross entropy loss over a two-dimensional tensor, balanced by weights
        :param y_pred: Predicted distribution
        :param y: Actual distribution
        :param v: Function that returns the weights for a given pixel
        :return: Rebalanced cross entropy loss
        """
        weights = v(y)
        return -torch.sum((nn.functional.log_softmax(y_pred, dim=1)) * y * weights) / (
                    y.shape[0] * y.shape[2] * y.shape[3])

    def save(self):
        """
        Save the model to the checkpoint folder
        :return: None
        """
        torch.save({
            "epochs": self.epochs,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }, f"CHECKPOINTS/{self.name}_{self.epochs}.pt")

    def load(self, name: str):
        """
        Load a model from the checkpoint folder
        :param name: Filename of the model
        :return: None
        """
        checkpoint = torch.load(f"CHECKPOINTS/{name}.pt")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.epochs = checkpoint["epochs"]

    def encode(self, image: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encode an LAB image into L and a distribution over Q values
        :param image: LAB image
        :return: L channel and distribution over Q values
        """
        image = image.to(self.device)
        l = image[:, :1, :, :]

        # Downsample the AB channels to 1/4 the size of the L channel as this is the output size of the model
        shape = (l.shape[-2] // 4, l.shape[-1] // 4)
        ab = torch.nn.functional.interpolate(image[:, 1:, :, :], size=shape, mode="bilinear")

        z = self.h_inv(ab)
        return l, z

    def decode(self, l: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decode an L channel and distribution over Q values into a LAB image
        :param l: L channel
        :param z: Distribution over Q values
        :return: LAB image
        """
        ab = self.h(z)
        # Upsample the AB channels to the size of the L channel
        ab = torch.nn.functional.interpolate(ab, size=l.shape[-2:], mode="bilinear")
        image = torch.cat((l, ab), dim=1)
        return image

    @staticmethod
    def lab_to_rgb(image: torch.Tensor) -> np.ndarray:
        """
        Convert a LAB image to RGB
        :param image: LAB image
        :return: RGB image
        """
        return np.clip(color.lab2rgb(image.cpu().permute(0, 2, 3, 1)), 0, 1)

    def plot(self, show: bool = True):
        """
        Plot the training and validation loss and safe it to the graphs folder
        :param show: true if the plot should be shown
        :return: None
        """
        plt.figure(figsize=(15, 5))

        # Plot a running average of the loss to smooth out the noise
        batches = np.linspace(0, self.epochs, len(self.train_loss))
        smoothed = np.convolve(self.train_loss, np.ones(100) / 100, mode="valid")
        plt.plot(batches[49:-50], smoothed, label="Training loss (running average over 100 batches)")

        batches = np.linspace(0, self.epochs, len(self.val_loss))
        smoothed = np.convolve(self.val_loss, np.ones(5) / 5, mode="valid")
        plt.plot(batches[2:-2], smoothed, label="Validation loss (running average over 5 epochs)")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"GRAPHS/{self.name}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        else:
            plt.close()

    def train_model(self, train: torch.utils.data.DataLoader, val: torch.utils.data.DataLoader, epochs: int):
        """
        Train the model, plot the loss and save checkpoints
        :param train: Training data loader
        :param val: Validation data loader
        :param epochs: Number of epochs to train for
        :return: None
        """
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

            # Every epoch, evaluate the model on the validation set and print the results
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

    def test(self, test: torch.utils.data.DataLoader) -> (float, float, float, float):
        """
        Test the model and calculate the evaluation metrics
        :param test: Test data loader
        :return: Mean squared error over the AB channels, Mean squared error over the RGB channels,
        peak signal-to-noise ratio and structural similarity index
        """
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

            return rmse_ab_metric, rmse_metric, psnr_metric, ssim_metric

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict the AB channels of an image from the L channel
        :param image: Image to predict for
        :return: Predicted image
        """
        self.eval()
        l = image[:, :1, :, :].to(self.device)
        with torch.no_grad():
            z = self(l)

        return self.decode(l, z)

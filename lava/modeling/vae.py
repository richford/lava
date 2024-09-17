import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class CSVDataset(Dataset):
    """
    A custom Dataset class for loading and preprocessing data from a CSV file.

    Attributes:
    csv_file (str or Path): The path to the CSV file containing the data.
    target_column (str, optional): The name of the column to use as the target variable. If None, the target variable is not used. Default is None.
    transform (callable, optional): A function/transform to apply to the input data. Default is None.

    Methods:
    __init__(self, csv_file, target_column=None, transform=None): Initializes the CSVDataset object.
    __len__(self): Returns the total number of samples in the dataset.
    __getitem__(self, idx): Returns a sample from the dataset at the specified index.
    """

    def __init__(self, csv_file, target_column=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column
        self.transform = transform

        if self.target_column:
            self.targets = self.data.pop(self.target_column).values
        else:
            self.targets = None

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the specified index.

        Parameters:
        idx (int): The index of the sample to retrieve.

        Returns:
        torch.Tensor: The input data sample as a PyTorch tensor.
        torch.Tensor (optional): The target variable as a PyTorch tensor. If the target_column is None, this value is not returned.
        """
        sample = self.data.iloc[idx].values.astype(np.float32)
        if self.targets is not None:
            target = self.targets[idx].astype(np.float32)
            return torch.tensor(sample), torch.tensor(target)
        return torch.tensor(sample)


# Function to create the DataLoader
def get_dataloader_from_csv(csv_file, batch_size=128, target_column=None):
    """
    Creates a DataLoader object for loading and batching data from a CSV file.

    Parameters:
    csv_file (str or Path): The path to the CSV file containing the data.
    batch_size (int, optional): The number of samples per batch. Default is 128.
    target_column (str, optional): The name of the column to use as the target variable. If None, the target variable is not used. Default is None.

    Returns:
    DataLoader: A DataLoader object that can be used to iterate over the dataset in batches.
    """
    dataset = CSVDataset(csv_file=csv_file, target_column=target_column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stopping to stop the training when the validation loss doesn't improve after
        a certain number of epochs (patience).

        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) model for dimensionality reduction and data generation.

    Attributes:
    latent_dim (int): The dimensionality of the latent space.
    min_val (float): The minimum value for the mean of the latent space.
    max_val (float): The maximum value for the mean of the latent space.
    min_val_logvar (float): The minimum value for the log variance of the latent space.
    max_val_logvar (float): The maximum value for the log variance of the latent space.
    fc1 (nn.Linear): The first fully connected layer of the encoder.
    fc2_mu (nn.Linear): The second fully connected layer of the encoder for the mean of the latent space.
    fc2_logvar (nn.Linear): The second fully connected layer of the encoder for the log variance of the latent space.
    fc3 (nn.Linear): The first fully connected layer of the decoder.
    fc4 (nn.Linear): The second fully connected layer of the decoder.

    Methods:
    __init__(self, input_dim=90, hidden_dim=400, latent_dim=2, min_val=-1.0, max_val=1.0, min_val_logvar=-10.0, max_val_logvar=10.0):
        Initializes the VAE model with the specified parameters.

    encode(self, x):
        Encodes the input data into the mean and log variance of the latent space.

    reparameterize(self, mu, logvar):
        Reparameterizes the latent space using the mean and log variance.

    decode(self, z):
        Decodes the latent space representation back into the original data space.

    forward(self, x, mask=None):
        Performs the forward pass of the VAE model, encoding, reparameterizing, and decoding the input data.
    """

    def __init__(
        self,
        input_dim=90,
        hidden_dim=45,
        latent_dim=2,
        min_val=-1.0,
        max_val=1.0,
        min_val_logvar=-10.0,
        max_val_logvar=10.0,
    ):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.min_val = min_val
        self.max_val = max_val
        self.min_val_logvar = min_val_logvar
        self.max_val_logvar = max_val_logvar

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)

        # Restricting the mu and logvar within the specified min and max values
        mu = torch.clamp(mu, self.min_val, self.max_val)
        logvar = torch.clamp(logvar, self.min_val_logvar, self.max_val_logvar)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Prevent the standard deviation from becoming too small which can
        # result in NaN values when multiplying by random noise during
        # reparameterization.
        std = torch.clamp(std, min=1e-6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def isotropy_regularization(mu, logvar):
    """
    Calculates the regularization term to penalize the covariance deviation from isotropy (identity matrix) in the latent space.

    Parameters:
    mu (torch.Tensor): The mean of the latent space. It should be of shape (batch_size, latent_dim).
    logvar (torch.Tensor): The log variance of the latent space. It should be of shape (batch_size, latent_dim).

    Returns:
    torch.Tensor: The regularization term for isotropy. It represents the sum of squared differences from 1 for the variance and the sum of squared mean values.
    """
    var = torch.exp(logvar)
    isotropy_penalty = torch.sum((var - 1) ** 2) + torch.sum(mu**2)
    return isotropy_penalty


def unit_variance_penalty(logvar):
    return torch.sum((torch.exp(logvar) - 1) ** 2)


def orthogonality_penalty(z):
    z_centered = z - z.mean(dim=0)
    cov_matrix = torch.mm(z_centered.T, z_centered) / z.size(0)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    return torch.sum(off_diag**2)


def monotonicity_penalty(recon_x, z):
    # Calculate differences in latent variables (along the first axis or between dimensions)
    latent_diff = z[:, 1:] - z[:, :-1]

    # Calculate differences in outputs (along the first axis or between dimensions)
    output_diff = recon_x[:, 1:] - recon_x[:, :-1]

    # Penalize where latent differences are positive but output differences are negative (violation of monotonicity)
    violation = torch.relu(-output_diff * latent_diff)

    # The penalty is proportional to the degree of monotonicity violation
    penalty = torch.sum(violation)

    return torch.sum(penalty)


def loss_function(
    recon_x,
    x,
    mu,
    logvar,
    z,
    mask,
    model,
    kl_weight=1.0,
    isotropy_weight=1.0,
    l1_weight=0.001,
    monotonicity_weight=0.0,
    orthogonality_weight=0.0,
    unit_variance_weight=0.0,
):
    """
    Calculates the total loss for the Variational Autoencoder (VAE) model.

    The loss is composed of three parts: Binary Cross Entropy (BCE), Kullback-Leibler Divergence (KLD),
    and L1 Regularization.

    Parameters:
    recon_x (torch.Tensor): The reconstructed input data.
    x (torch.Tensor): The original input data.
    mu (torch.Tensor): The mean of the latent space.
    logvar (torch.Tensor): The log variance of the latent space.
    mask (torch.Tensor): A binary mask indicating the valid data points.
    model (nn.Module): The VAE model.
    isotropy_weight (float, optional): The weight for isotropy regularization. Default is 1.0.
    kl_wieght (float, optional): The weight for KL Divergence. Default is 1.0.
    l1_weight (float, optional): The weight for L1 regularization. Default is 0.001.

    Returns:
    torch.Tensor: The total loss for the VAE model.
    """
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="none")
    loss = (BCE * mask).sum()

    # KL Divergence
    loss += -0.5 * kl_weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # L1 Regularization
    loss += l1_weight * sum(torch.sum(torch.abs(param)) for param in model.parameters())

    # Add isotropy regularization term
    loss += isotropy_weight * isotropy_regularization(mu, logvar)

    # Monotonicity regularization term
    loss += monotonicity_weight * monotonicity_penalty(recon_x, z)

    # Orthogonality regularization term
    loss += orthogonality_weight * orthogonality_penalty(z)

    # Unit variance regularization term
    loss += unit_variance_weight * unit_variance_penalty(logvar)

    return loss


def create_mask(data):
    """
    Creates a binary mask and replaces NaN values in the input data with zeros.

    Parameters:
    data (torch.Tensor): The input data tensor. It should be of shape (batch_size, features).

    Returns:
    torch.Tensor: The modified data tensor with NaN values replaced by zeros.
    torch.Tensor: The binary mask tensor indicating the valid data points (1) and NaN values (0).
    """
    mask = ~torch.isnan(data)
    data = torch.where(mask, data, torch.zeros_like(data))  # Replace NaNs with 0s
    return data, mask.float()


def train(
    model,
    train_loader,
    optimizer,
    epoch,
    kl_weight=1.0,
    isotropy_weight=1.0,
    l1_weight=0.001,
    monotonicity_weight=0.0,
    orthogonality_weight=0.0,
    unit_variance_weight=0.0,
    device="cpu",
    verbose=True,
):
    """
    Trains the Variational Autoencoder (VAE) model for a single epoch.

    Parameters:
    model (nn.Module): The VAE model to be trained.
    train_loader (DataLoader): The DataLoader object providing the training data in batches.
    optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    epoch (int): The current epoch number.
    kl_weight (float, optional): The weight for KL Divergence. Default is 1.0.
    isoropy_weight (float, optional): The weight for isotropy regularization. Default is 1.0.
    l1_weight (float, optional): The weight for L1 regularization. Default is 0.001.
    device (str, optional): The device to run the training on. Default is "cpu".
    verbase (bool, optional): Whether to print progress updates during training. Default is False.

    Returns:
    None
    """
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        data, mask = create_mask(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(
            recon_x=recon_batch,
            x=data,
            mu=mu,
            z=z,
            logvar=logvar,
            mask=mask,
            model=model,
            isotropy_weight=isotropy_weight,
            kl_weight=kl_weight,
            l1_weight=l1_weight,
            monotonicity_weight=monotonicity_weight,
            orthogonality_weight=orthogonality_weight,
            unit_variance_weight=unit_variance_weight,
        )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    if verbose:
        logger.info(f"====> Epoch: {epoch} Average loss: {train_loss:.4f}")

    return train_loss


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    max_epochs=100,
    lr_plateau_patience=3,
    early_stopping_patience=5,
    min_delta=0.001,
    kl_weight=1.0,
    isotropy_weight=1.0,
    l1_weight=0.001,
    monotonicity_weight=0.0,
    orthogonality_weight=0.0,
    unit_variance_weight=0.0,
    device="cpu",
    verbose=True,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=lr_plateau_patience,
    )
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=min_delta)

    for epoch in range(1, max_epochs + 1):
        train_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            kl_weight=kl_weight,
            isotropy_weight=isotropy_weight,
            l1_weight=l1_weight,
            device=device,
            verbose=False,
        )

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch, mask = create_mask(batch)
                recon_batch, mu, logvar, z = model(batch)
                loss = loss_function(
                    recon_x=recon_batch,
                    x=batch,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    mask=mask,
                    model=model,
                    isotropy_weight=isotropy_weight,
                    kl_weight=kl_weight,
                    l1_weight=l1_weight,
                    monotonicity_weight=monotonicity_weight,
                    orthogonality_weight=orthogonality_weight,
                    unit_variance_weight=unit_variance_weight,
                )
                val_loss += loss.item()

        # Average losses over the dataset
        val_loss /= len(val_loader.dataset)

        if verbose:
            logger.info(
                f"Epoch {epoch}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.1e}\n"
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                logger.info("Early stopping triggered.")
            break

        scheduler.step(val_loss)

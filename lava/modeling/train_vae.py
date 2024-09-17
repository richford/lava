import os
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from lava.config import CHECKPOINT_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from lava.modeling.cross_validation import load_checkpoint, nested_cross_validation
from lava.modeling.vae import VAE, fit, get_dataloader_from_csv, weights_init

app = typer.Typer()


@app.command()
def main(
    mode: str = "train",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "model.pth",
    hyperparam_path: Path = MODELS_DIR / "best_hyperparams.npy",
    use_mps: bool = False,
    random_seed: int = 1729,
):
    """
    Trains a Variational Autoencoder (VAE) model using the provided CSV file of features.

    Parameters:
    features_path (Path): The path to the CSV file containing the features data. Default is "PROCESSED_DATA_DIR/features.csv".
    model_path (Path): The path to save the trained VAE model. Default is "MODELS_DIR/model.pth".

    Returns:
    None
    """
    # Create DataLoader from CSV
    batch_size = 128
    train_loader = get_dataloader_from_csv(
        csv_file=train_path, batch_size=batch_size, target_column=None
    )

    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and use_mps:
        device_str = "mps"
    else:
        device_str = "cpu"

    device = torch.device(device_str)

    if mode == "optimize":
        nested_cross_validation(
            train_loader,
            random_seed=random_seed,
            n_splits=5,
            n_repeats=20,
            n_trials=100,
            device=device,
        )
    elif mode == "train":
        _, best_hyperparams, _ = load_checkpoint(fold=0, checkpoint_dir=CHECKPOINT_DIR)

        if best_hyperparams is None:
            best_hyperparams = {
                "latent_dim": 2,
                "hidden_dim": 32,
                "lr": 1e-3,
                "l1_weight": 1e-4,
                "kl_weight": 1.0,
                "isotropy_weight": 1e-3,
                "monotonicity_weight": 0.0,
                "orthogonality_weight": 1e-3,
                "unit_variance_weight": 0.0,
            }

        model = VAE(
            input_dim=train_loader.dataset[0].shape[0],
            hidden_dim=best_hyperparams["hidden_dim"],
            latent_dim=best_hyperparams["latent_dim"],
            min_val=-1.0,
            max_val=1.0,
        ).to(device)

        model.apply(weights_init)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams["lr"])

        rng = np.random.RandomState(seed=random_seed)
        shuffle_split = ShuffleSplit(test_size=0.2, random_state=rng)
        train_idx, val_idx = next(shuffle_split.split(train_loader.dataset))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            train_loader.dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=val_sampler)

        max_epochs = 100

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            max_epochs=max_epochs,
            lr_plateau_patience=3,
            early_stopping_patience=8,
            min_delta=0,
            kl_weight=best_hyperparams["kl_weight"],
            isotropy_weight=best_hyperparams["isotropy_weight"],
            l1_weight=best_hyperparams["l1_weight"],
            monotonicity_weight=best_hyperparams["monotonicity_weight"],
            orthogonality_weight=best_hyperparams["orthogonality_weight"],
            unit_variance_weight=best_hyperparams["unit_variance_weight"],
            device=device,
            verbose=True,
        )

        # Save the trained model
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        torch.save(model.state_dict(), model_path)
        logger.info("Model saved successfully.")


if __name__ == "__main__":
    app()

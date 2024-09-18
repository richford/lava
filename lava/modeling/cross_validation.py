import json
import os

import numpy as np
import optuna
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from lava.config import CHECKPOINT_DIR
from lava.dataset import get_total_score_bins
from lava.modeling.vae import VAE, create_mask, fit, loss_function, weights_init


def validate_cv_params(current_cv_params, saved_cv_params):
    """Check if the current CV parameters match the saved ones."""
    for param, value in current_cv_params.items():
        if saved_cv_params.get(param) != value:
            return False
    return True


def save_checkpoint(
    fold, cv_params, model, best_hyperparams, metrics, test_score, checkpoint_dir=CHECKPOINT_DIR
):
    """
    Save the model, best hyperparameters, and test scores for a completed fold.
    Args:
        fold: The fold number.
        cv_params: The CV parameters.
        model: The trained model.
        best_hyperparams: The best hyperparameters found during inner CV.
        metrics: The predictive performance metrics for this fold.
        test_score: The test score for this fold.
    """
    # Define paths to save checkpoint files
    model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")
    results_path = os.path.join(checkpoint_dir, "results.json")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save the model state dict
    torch.save(model.state_dict(), model_path)

    # Load existing results if checkpoint exists
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Update results dictionary
    results["cv_params"] = cv_params
    results[f"cv_{str(fold)}"] = {
        "best_hyperparams": best_hyperparams,
        "test_score": test_score,
        "metrics": metrics,
    }

    # Save the results dictionary to a JSON file
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def load_cv_params(checkpoint_dir=CHECKPOINT_DIR):
    """Load CV parameters from the checkpoint directory.

    Args:
        checkpoint_dir: The directory containing the checkpoint files. Default is "CHECKPOINT_DIR".
    Returns:
        cv_params: A dictionary containing the CV parameters.
    """
    results_path = os.path.join(checkpoint_dir, "results.json")

    # Load hyperparameters and test score from the results JSON
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    return results.get("cv_params", {})


def load_checkpoint(fold, input_dim, checkpoint_dir=CHECKPOINT_DIR):
    """
    Load the model and results from a specific fold checkpoint.
    Args:
        fold: The fold number to load.
    Returns:
        model: The model loaded from the checkpoint.
        best_hyperparams: The best hyperparameters for this fold.
        test_score: The test score for this fold.
    """
    # Define paths to save checkpoint files
    model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")
    results_path = os.path.join(checkpoint_dir, "results.json")

    # Load hyperparameters and test score from the results JSON
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    if (f"cv_{str(fold)}" in results) and os.path.exists(model_path):
        # Load the model state dict
        best_hyperparams = results[f"cv_{str(fold)}"]["best_hyperparams"]
        test_score = results[f"cv_{str(fold)}"]["test_score"]

        model = VAE(
            input_dim=input_dim,
            hidden_dim=best_hyperparams["hidden_dim"],
            latent_dim=best_hyperparams["latent_dim"],
            min_val=-1.0,
            max_val=1.0,
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))

        return model, best_hyperparams, test_score
    else:
        return None, None, None


# Function to evaluate the model on the validation set
def evaluate_model(
    model,
    validation_loader,
    l1_weight,
    kl_weight,
    isotropy_weight,
    device,
    monotonicity_weight=0.0,
    orthogonality_weight=0.0,
    unit_variance_weight=0.0,
):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for data in validation_loader:
            data = data.to(device)
            data, mask = create_mask(data)
            recon_batch, mu, logvar, z = model(data)
            loss = loss_function(
                recon_x=recon_batch,
                x=data,
                mu=mu,
                logvar=logvar,
                z=z,
                mask=mask,
                model=model,
                l1_weight=l1_weight,
                kl_weight=kl_weight,
                isotropy_weight=isotropy_weight,
                monotonicity_weight=monotonicity_weight,
                orthogonality_weight=orthogonality_weight,
                unit_variance_weight=unit_variance_weight,
            )
            validation_loss += loss.item()

    validation_loss /= len(validation_loader.dataset)
    return validation_loss


def predictive_performance(model, test_loader):
    """
    Evaluate the performance of a model on a test dataset and save metrics to a file.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader providing test data.
        save_path: Path to save the evaluation metrics as a JSON file.

    Returns:
        metrics: A dictionary containing the computed metrics.
    """
    model.eval()  # Set model to evaluation mode

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            batch, mask = create_mask(batch)
            probs, _, _, _ = model(batch)
            preds = (probs >= 0.5).int()  # Binarize predictions

            # Flatten the batch, probs, preds, and mask
            mask = mask.flatten().cpu().numpy().astype(int)
            probs = probs.flatten().cpu().numpy()
            batch = batch.flatten().cpu().numpy()
            preds = preds.flatten().cpu().numpy()

            all_labels.extend(batch[mask].astype(int))
            all_preds.extend(preds[mask].astype(int))
            all_probs.extend(probs[mask])

    # Convert lists to numpy arrays for metric computation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    if fp + tn == 0:
        false_positive_rate = np.nan
    else:
        false_positive_rate = fp / (fp + tn)

    # If only one class is present in the labels, then ROC AUC score is not defined.
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, labels=[0, 1])
    except ValueError:
        roc_auc = np.nan

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, labels=[0, 1], zero_division=np.nan)
    recall = recall_score(all_labels, all_preds, labels=[0, 1], zero_division=np.nan)
    f1 = f1_score(all_labels, all_preds, labels=[0, 1], zero_division=np.nan)
    log_loss_score = log_loss(all_labels, all_probs, labels=[0, 1])  # Log Loss (Cross-entropy)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_denom == 0:
        mcc = np.nan
    else:
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom

    brier_score = brier_score_loss(all_labels, all_probs, pos_label=1)  # Brier score
    cohen_kappa = cohen_kappa_score(all_labels, all_preds, labels=[0, 1])  # Cohen's Kappa

    # Compute metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "log_loss": log_loss_score,
        "matthews_corrcoef": mcc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "false_positive_rate": false_positive_rate,
        "brier_score_loss": brier_score,
        "cohens_kappa": cohen_kappa,
    }

    return metrics


def nested_cross_validation(
    data_loader,
    qbins,
    n_splits=5,
    n_repeats=20,
    n_trials=50,
    device="cpu",
    checkpoint_dir=CHECKPOINT_DIR,
    random_seed=1729,
):
    test_size = 0.2
    max_epochs = 100
    lr_plateau_patience = 3
    early_stopping_patience = 5
    min_delta = 0.001
    batch_size = 128

    # Create score bins for stratified KFold cross-validation
    bins = get_total_score_bins(data_loader.dataset.data, qbins)
    logger.debug(f"Score bins: {bins}")

    current_cv_params = {
        "qbins": qbins,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_seed": random_seed,
        "test_size": test_size,
        "batch_size": batch_size,
        "latent_dim": 2,
        "fit_params": {
            "max_epochs": max_epochs,
            "lr_plateau_patience": lr_plateau_patience,
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
        },
    }

    rng = np.random.RandomState(seed=random_seed)
    outer_cv = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=rng)
    models = []
    test_scores = []
    best_hyperparams = []
    input_dim = data_loader.dataset[0].shape[0]

    saved_cv_params = load_cv_params(checkpoint_dir=checkpoint_dir)

    # Check if the saved CV params match the current params
    if not validate_cv_params(current_cv_params, saved_cv_params):
        logger.info("Cross-validation parameters do not match the saved checkpoints.")
        logger.info("Invalidating the existing checkpoints.")

        # Optionally delete the checkpoint files
        for file in os.listdir(checkpoint_dir):
            if file.startswith("model_fold_") or file.startswith("results.json"):
                os.remove(os.path.join(CHECKPOINT_DIR, file))

    # First split the data into training and validation sets
    # The validation set is used to tune hyperparameters
    for fold, (train_test_idx, val_idx) in enumerate(outer_cv.split(data_loader.dataset, bins)):
        model, hyperparams, test_score = load_checkpoint(
            fold=fold, input_dim=input_dim, checkpoint_dir=checkpoint_dir
        )

        if all([model is not None, hyperparams is not None, test_score is not None]):
            models.append(model)
            best_hyperparams.append(hyperparams)
            test_scores.append(test_score)
            continue

        train_test_sampler = SubsetRandomSampler(train_test_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_test_loader = DataLoader(
            data_loader.dataset, batch_size=batch_size, sampler=train_test_sampler
        )
        val_loader = DataLoader(data_loader.dataset, batch_size=batch_size, sampler=val_sampler)

        shuffle_split = StratifiedShuffleSplit(test_size=test_size, random_state=rng)

        def objective(trial):
            hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            isotropy_weight = trial.suggest_float("isotropy_weight", 1e-5, 1e-2, log=True)
            orthogonality_weight = trial.suggest_float(
                "orthogonality_weight", 1e-4, 1e-1, log=True
            )
            l1_weight = trial.suggest_float("l1_weight", 1e-7, 1e-2, log=True)

            # For now, hard code the monotonicity, KL divergence, and unit variance weights
            #
            # This is what we should use if we want to optimize these weights
            #
            # monotonicity_weight = trial.suggest_float("monotonicity_weight", 1e-6, 1e-3, log=True)
            # kl_weight = trial.suggest_float("kl_weight", 1.0, 10.0, log=True)
            # unit_variance_weight = trial.suggest_float(
            #     "unit_variance_weight", 1e-5, 1e-2, log=True
            # )
            #
            # But for now, we will use fixed weights
            monotonicity_weight = trial.suggest_float("monotonicity_weight", 0.0, 0.0)
            kl_weight = trial.suggest_float("kl_weight", 1.0, 1.0)
            unit_variance_weight = trial.suggest_float("unit_variance_weight", 0.0, 0.0)
            latent_dim = trial.suggest_int(
                "latent_dim", current_cv_params["latent_dim"], current_cv_params["latent_dim"]
            )

            model = VAE(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                min_val=-1.0,
                max_val=1.0,
            ).to(device)

            model.apply(weights_init)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Split the training set into training and test sets
            # The test set is used to for early stopping and the the learning rate scheduler
            train_test_bins = get_total_score_bins(train_test_loader.dataset.data, qbins)
            train_idx, test_idx = next(
                shuffle_split.split(train_test_loader.dataset, train_test_bins)
            )

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            train_loader = DataLoader(
                train_test_loader.dataset, batch_size=batch_size, sampler=train_sampler
            )
            test_loader = DataLoader(
                train_test_loader.dataset, batch_size=batch_size, sampler=test_sampler
            )

            fit(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                max_epochs=max_epochs,
                lr_plateau_patience=lr_plateau_patience,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
                kl_weight=kl_weight,
                isotropy_weight=isotropy_weight,
                l1_weight=l1_weight,
                monotonicity_weight=monotonicity_weight,
                orthogonality_weight=orthogonality_weight,
                unit_variance_weight=unit_variance_weight,
                device=device,
                verbose=False,
            )

            validation_loss = evaluate_model(
                model=model,
                validation_loader=val_loader,
                l1_weight=l1_weight,
                kl_weight=kl_weight,
                isotropy_weight=isotropy_weight,
                monotonicity_weight=monotonicity_weight,
                orthogonality_weight=orthogonality_weight,
                unit_variance_weight=unit_variance_weight,
                device=device,
            )
            return validation_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

        best_hyperparams.append(study.best_params)

        model = VAE(
            input_dim=input_dim,
            hidden_dim=study.best_params["hidden_dim"],
            latent_dim=study.best_params["latent_dim"],
            min_val=-1.0,
            max_val=1.0,
        ).to(device)

        model.apply(weights_init)

        optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params["lr"])

        # Split the training set into training and test sets
        # The test set is used to for early stopping and the the learning rate scheduler
        train_test_bins = get_total_score_bins(train_test_loader.dataset.data, qbins)
        train_idx, test_idx = next(shuffle_split.split(train_test_loader.dataset, train_test_bins))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_test_loader.dataset, batch_size=batch_size, sampler=train_sampler
        )
        test_loader = DataLoader(
            train_test_loader.dataset, batch_size=batch_size, sampler=test_sampler
        )

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            max_epochs=max_epochs,
            lr_plateau_patience=lr_plateau_patience,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            kl_weight=study.best_params["kl_weight"],
            isotropy_weight=study.best_params["isotropy_weight"],
            l1_weight=study.best_params["l1_weight"],
            monotonicity_weight=study.best_params["monotonicity_weight"],
            orthogonality_weight=study.best_params["orthogonality_weight"],
            unit_variance_weight=study.best_params["unit_variance_weight"],
            device=device,
            verbose=False,
        )

        val_loss = evaluate_model(
            model=model,
            validation_loader=val_loader,
            l1_weight=study.best_params["l1_weight"],
            kl_weight=study.best_params["kl_weight"],
            isotropy_weight=study.best_params["isotropy_weight"],
            monotonicity_weight=study.best_params["monotonicity_weight"],
            orthogonality_weight=study.best_params["orthogonality_weight"],
            unit_variance_weight=study.best_params["unit_variance_weight"],
            device=device,
        )

        test_scores.append(val_loss)

        metrics = predictive_performance(model, val_loader)

        save_checkpoint(
            fold=fold,
            cv_params=current_cv_params,
            model=model,
            best_hyperparams=study.best_params,
            metrics=metrics,
            test_score=val_loss,
            checkpoint_dir=checkpoint_dir,
        )

    return models, best_hyperparams, test_scores

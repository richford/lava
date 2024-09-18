import string
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from lava.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def get_total_score_bins(X: pd.DataFrame, bins: List[int]) -> np.ndarray:
    # Step 1: Compute total score for each response chain (summing across the 90 elements)
    total_scores = X.sum(axis="columns", skipna=True)

    # Step 2: Use the input qbins to bin by quantile. Remove the min and max bin thresholds.
    bins = np.digitize(total_scores, bins=bins) - 1

    return bins


def undersample(df: pd.DataFrame, random_state: any, q: int) -> pd.DataFrame:
    # Step 1: Compute total score for each response chain (summing across the 90 elements)
    total_scores = df.sum(axis="columns", skipna=True)
    max_score = len(df.columns) - 1  # Exclude the target column

    # Step 2: Identify perfect scorers
    perfect_scorers_mask = total_scores == max_score

    # Separate the perfect scorers from the non-perfect scorers
    df_perfect = df[perfect_scorers_mask]
    df_non_perfect = df[~perfect_scorers_mask]

    logger.info(f"Resampling runs...")

    # Apply undersampling to the perfect scorers
    # Keep this fraction of the perfect scorers
    undersample_ratio = 0.4
    df_perfect_resampled = df_perfect.sample(
        frac=undersample_ratio,
        replace=False,
        random_state=random_state,
        axis="index",
        ignore_index=True,
    )

    logger.info(
        f"Found {df_perfect.shape[0]} / {df.shape[0]} perfect scores. Removed {df_perfect.shape[0] - df_perfect_resampled.shape[0]}"
    )

    # Combine the undersampled perfect scorers with the non-perfect scorers
    df_resampled = pd.concat(
        [df_perfect_resampled, df_non_perfect], axis="index", ignore_index=True
    )

    # Finally shuffle the dataset
    df_resampled = df_resampled.sample(
        frac=1, replace=False, random_state=random_state, axis="index", ignore_index=True
    )

    logger.info(f"Original dataset size: {df.shape[0]}")
    logger.info(f"Resampled dataset size: {df_resampled.shape[0]}")

    total_scores = df_resampled.sum(axis="columns", skipna=True)
    _, qbins = pd.qcut(total_scores, q=q, retbins=True)

    qbins = np.round(qbins).astype(int)

    logger.info(f"Quintile bins for resampled data: {qbins}")

    return df_resampled, qbins


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "letter.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    random_seed: int = 1729,
):
    logger.info("Processing dataset...")

    letter_columns = list(string.ascii_letters) + [
        "a_Long",
        "a_Short",
        "b_Sound",
        "c_Hard",
        "c_Soft",
        "ch_Sound",
        "d_Sound",
        "e_Long",
        "e_Short",
        "f_Sound",
        "g_Hard",
        "g_Soft",
        "h_Sound",
        "i_Long",
        "i_Short",
        "j_Sound",
        "k_Sound",
        "l_Sound",
        "m_Sound",
        "n_Sound",
        "o_Long",
        "o_Short",
        "oo_Long",
        "oo_Short",
        "p_Sound",
        "qu_Sound",
        "r_Sound",
        "s_Sound",
        "sh_Sound",
        "t_Sound",
        "th_Unvoiced",
        "u_Long",
        "u_Short",
        "v_Sound",
        "w_Sound",
        "x_Sound",
        "y_Sound",
        "z_Sound",
    ]

    letter_trials = pd.read_csv(input_path, usecols=letter_columns)

    rng = np.random.RandomState(seed=random_seed)

    # Undersample the perfect scorers
    letter_trials, qbins = undersample(letter_trials, random_state=rng, q=5)

    # Split into training (60%), test (20%), and report (20%) sets
    train_data, temp_data = train_test_split(letter_trials, test_size=0.4, random_state=rng)
    test_data, report_data = train_test_split(temp_data, test_size=0.5, random_state=rng)

    # Save the splits
    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    test_data.to_csv(f"{output_dir}/test.csv", index=False)
    report_data.to_csv(f"{output_dir}/report.csv", index=False)

    # Save the quantile bins
    np.savetxt(f"{output_dir}/qbins.csv", qbins, delimiter=",", fmt="%d")

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

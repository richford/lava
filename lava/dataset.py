import string
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from lava.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


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

    # Split into training (60%), test (20%), and report (20%) sets
    rng = np.random.RandomState(seed=random_seed)
    train_data, temp_data = train_test_split(letter_trials, test_size=0.4, random_state=rng)
    test_data, report_data = train_test_split(temp_data, test_size=0.5, random_state=rng)

    # Save the splits
    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    test_data.to_csv(f"{output_dir}/test.csv", index=False)
    report_data.to_csv(f"{output_dir}/report.csv", index=False)

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

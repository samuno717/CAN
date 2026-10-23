import numpy as np
import pandas as pd
from loguru import logger


def preprocess_and_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info("Begin preprocessing and data splitting...")

    if "timestamp" in df.columns:
        df = df.drop("timestamp", axis=1)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["can_id"], prefix="can_id", dtype=np.float32)

    total_len = len(df)

    # Points of split
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    # Split
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    test_ratio = 1.0 - (train_ratio + val_ratio)

    logger.success(
        f"Split done: "
        f"{train_ratio:.0%} Train ({len(df_train)} rows) | "
        f"{val_ratio:.0%} Val ({len(df_val)} rows) | "
        f"{test_ratio:.0%} Test ({len(df_test)} rows)"
    )

    return df_train, df_val, df_test

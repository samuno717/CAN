import pandas as pd
import numpy as np
from loguru import logger

def preprocess_and_split(df: pd.DataFrame, split_ratio: float):
    logger.info("Begin preprocessing and data splitting...")

    if "timestamp" in df.columns:
        df = df.drop("timestamp", axis=1)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["can_id"], prefix="can_id", dtype=np.float32)

    # Temporal Split
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    logger.success(
        f"Split done: {split_ratio:.0%} Train ({len(df_train)} rows) | "
        f"{1 - split_ratio:.0%} Val ({len(df_val)} rows)"
    )
    return df_train, df_val

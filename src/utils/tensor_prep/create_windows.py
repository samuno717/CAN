from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


def create_windows(data_matrix, sequence_length: int) -> np.ndarray:
    num_samples = data_matrix.shape[0] - sequence_length + 1
    num_features = data_matrix.shape[1]

    # stride_bytes = (104, 4)
    # 104 bytes is the distance in RAM between each row
    # 4 bytes is the distance in RAM between each column in the same row
    stride_bytes = data_matrix.strides
    X_tensors = np.lib.stride_tricks.as_strided(
        data_matrix,
        shape=(num_samples, sequence_length, num_features),
        strides=(stride_bytes[0], # step between each window
                 stride_bytes[0], # step between each frame in the same window
                 stride_bytes[1]) # step between each column
    )
    logger.info(f"Created windows shape: {X_tensors.shape}")
    return X_tensors

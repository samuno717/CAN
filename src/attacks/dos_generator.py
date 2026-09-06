import numpy as np
from loguru import logger


def inject_dos_attack(
    clean_tensors_path,
    output_tensors_path,
    output_labels_path,
    start_row: int,
    duration_rows: int,
    target_id_col: int = 9,
):
    logger.info(f"Loading clean dataset from: {clean_tensors_path}")
    X = np.load(clean_tensors_path).copy()
    y = np.zeros(len(X), dtype=np.int8)

    end_row = start_row + duration_rows
    if end_row > len(X):
        raise ValueError(
            f"Attack window [{start_row}:{end_row}] exceeds dataset length ({len(X)})"
        )

    logger.warning(f"Injecting DoS sequence in rows: [{start_row} : {end_row}]")

    # 1. Payload: flood with 0xFF (1.0 in MinMax scaling)
    X[start_row:end_row, 0:8] = 1.0

    # 2. Time delta: minimal interval (bus saturation)
    X[start_row:end_row, 8] = 0.0

    # 3. ID arbitration: monopolize bus with high-priority ID
    X[start_row:end_row, 9:26] = 0.0
    X[start_row:end_row, target_id_col] = 1.0

    # 4. Binary Ground Truth labels
    y[start_row:end_row] = 1

    # Ensure target directory exists and save artifacts
    output_tensors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_tensors_path, X)
    np.save(output_labels_path, y)

    logger.success(f"Attacked dataset saved to: {output_tensors_path}")
    logger.success(f"Ground truth labels saved to: {output_labels_path}")

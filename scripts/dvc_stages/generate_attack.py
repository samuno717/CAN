import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from config import PROCESSED_ATTACKS_DIR, RAW_DATA_DIR
from src.utils.attacks.dos import inject_dos_df, load_test_raw_df
from src.utils.attacks.replay import inject_replay_df


def apply_attack(df: pd.DataFrame, attack_type: str, step_cfg: DictConfig | dict) -> pd.DataFrame:
    if attack_type == "dos":
        frame_str = f"{step_cfg['can_id_hex']}#{step_cfg['payload_hex']}"
        logger.info(f"Injecting DoS at row {step_cfg['start_row']} ({step_cfg['duration_rows']} frames)...")
        return inject_dos_df(
            df=df,
            start_row=int(step_cfg["start_row"]),
            dos_rows=int(step_cfg["duration_rows"]),
            attack_payload_str=frame_str,
            physical_delay=float(step_cfg["physical_delay"]),
        )

    if attack_type == "replay":
        logger.info(f"Injecting Replay at row {step_cfg['start_row']} ({step_cfg['duration_rows']} frames)...")
        return inject_replay_df(
            df=df,
            capture_start_row=int(step_cfg["capture_start_row"]),
            start_row=int(step_cfg["start_row"]),
            duration_rows=int(step_cfg["duration_rows"]),
            target_can_id_hex=str(step_cfg["target_can_id_hex"]),
            speed_factor=float(step_cfg["speed_factor"]),
            min_bus_gap=float(step_cfg["min_bus_gap"]),
        )

    raise ValueError(f"Unknown attack type: {attack_type}")


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    scenario = str(cfg.attacks.active_scenario)
    raw_log_path = RAW_DATA_DIR / cfg.parsing.raw_file

    df = load_test_raw_df(
        raw_data=raw_log_path,
        train_ratio=float(cfg.preprocessing.train_ratio),
        val_ratio=float(cfg.preprocessing.val_ratio),
    )

    if scenario == "multi":
        for step in cfg.attacks.multi.steps:
            df = apply_attack(df, str(step["type"]), step)
    else:
        df = apply_attack(df, scenario, cfg.attacks[scenario])

    PROCESSED_ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROCESSED_ATTACKS_DIR / f"{scenario}_attack.log"
    labels_path = PROCESSED_ATTACKS_DIR / f"{scenario}_labels.npy"

    np.save(labels_path, df["label"].to_numpy(dtype=np.int8))
    df.drop(columns=["label"]).to_csv(log_path, sep=" ", index=False, header=False)
    logger.success(f"[{scenario}] Saved {len(df):,} frames to {log_path.name}")


if __name__ == "__main__":
    main()

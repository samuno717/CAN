import hydra
from loguru import logger
from omegaconf import DictConfig
import pandas as pd

from config import ATTACK_NAMES, PROCESSED_ATTACKS_DIR, RAW_DATA_DIR
from src.utils.attacks import (
    inject_dos_df,
    inject_frequency_anomaly_df,
    inject_fuzzing_df,
    inject_payload_manipulation_df,
    inject_replay_df,
    inject_suspension_df,
    load_test_raw_df,
    save_attack_dataset,
)


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

    if attack_type == "payload":
        logger.info(
            f"Injecting Masquerade Payload Manipulation at row {step_cfg['start_row']} "
            f"(window={step_cfg['duration_rows']}, id={step_cfg['target_can_id_hex']}, mode={step_cfg['mode']})..."
        )
        return inject_payload_manipulation_df(
            df=df,
            start_row=int(step_cfg["start_row"]),
            duration_rows=int(step_cfg["duration_rows"]),
            target_can_id_hex=str(step_cfg["target_can_id_hex"]),
            mode=str(step_cfg["mode"]),
            target_bytes=list(step_cfg.get("target_bytes", range(8))),
            custom_payload_hex=str(step_cfg.get("custom_payload_hex", "FFFFFFFFFFFFFFFF")),
            drift_step=int(step_cfg.get("drift_step", 2)),
            seed=int(step_cfg.get("seed", 42)),
        )

    if attack_type == "frequency":
        logger.info(
            f"Injecting Frequency Anomaly at row {step_cfg['start_row']} "
            f"(window={step_cfg['duration_rows']}, id={step_cfg['target_can_id_hex']}, "
            f"multiplier={step_cfg['freq_multiplier']}x)..."
        )
        return inject_frequency_anomaly_df(
            df=df,
            start_row=int(step_cfg["start_row"]),
            duration_rows=int(step_cfg["duration_rows"]),
            target_can_id_hex=str(step_cfg["target_can_id_hex"]),
            freq_multiplier=int(step_cfg["freq_multiplier"]),
            min_bus_gap=float(step_cfg["min_bus_gap"]),
        )

    if attack_type == "fuzzing":
        logger.info(
            f"Injecting Fuzzing at row {step_cfg['start_row']} "
            f"({step_cfg['num_frames']} frames, id_mode={step_cfg['id_mode']})..."
        )
        return inject_fuzzing_df(
            df=df,
            start_row=int(step_cfg["start_row"]),
            num_frames=int(step_cfg["num_frames"]),
            id_mode=str(step_cfg["id_mode"]),
            interval_min=float(step_cfg["interval_min"]),
            interval_max=float(step_cfg["interval_max"]),
            min_bus_gap=float(step_cfg["min_bus_gap"]),
            seed=int(step_cfg.get("seed", 42)),
        )

    if attack_type == "suspension":
        logger.info(
            f"Injecting Suspension (Bus-Off) at row {step_cfg['start_row']} "
            f"(window={step_cfg['duration_rows']}, target_id={step_cfg['target_can_id_hex']})..."
        )
        return inject_suspension_df(
            df=df,
            start_row=int(step_cfg["start_row"]),
            duration_rows=int(step_cfg["duration_rows"]),
            target_can_id_hex=str(step_cfg["target_can_id_hex"]),
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
    clean_count = len(df)

    if scenario == "multi":
        for step in cfg.attacks.multi.steps:
            df = apply_attack(df, str(step["type"]), step)
    else:
        df = apply_attack(df, scenario, cfg.attacks[scenario])

    log_path = PROCESSED_ATTACKS_DIR / f"{scenario}_attack.log"
    labels_path = PROCESSED_ATTACKS_DIR / f"{scenario}_labels.npy"
    save_attack_dataset(df, log_path=log_path, labels_path=labels_path)

    label_counts = df["label"].value_counts().sort_index()
    breakdown = ", ".join(
        f"{ATTACK_NAMES.get(int(lbl), 'Normal')}: {cnt:,}"
        for lbl, cnt in label_counts.items()
    )
    logger.info(f"[{scenario}] Breakdown ({clean_count:,} base -> {len(df):,} total frames): {breakdown}")
    logger.success(f"[{scenario}] Saved {log_path.name} and {labels_path.name}")


if __name__ == "__main__":
    main()

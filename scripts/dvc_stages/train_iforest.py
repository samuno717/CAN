import hydra
from omegaconf import DictConfig

from config import FEATURES_TRAIN_PATH, WEIGHTS_DIR
from src.models.isolation_forest import (
    CANIsolationForest,
    extract_iforest_window_features,
)


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    seq_len = int(cfg.model.sequence_length)
    seq_step = int(cfg.model.sequence_step)
    feature_mode = str(cfg.iforest.feature_mode)
    n_estimators = int(cfg.iforest.n_estimators)
    max_samples = cfg.iforest.max_samples
    if isinstance(max_samples, str) and max_samples.isdigit():
        max_samples = int(max_samples)

    x_train = extract_iforest_window_features(
        file_path=FEATURES_TRAIN_PATH,
        seq_len=seq_len,
        stride=seq_step,
        feature_mode=feature_mode,
    )

    iforest = CANIsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=cfg.iforest.contamination,
        max_features=float(cfg.iforest.max_features),
        random_state=int(cfg.training.seed),
        n_jobs=-1,
        feature_mode=feature_mode,
    )
    iforest.fit(x_train)

    model_save_path = WEIGHTS_DIR / str(cfg.iforest.path)
    iforest.save(model_save_path)


if __name__ == "__main__":
    main()

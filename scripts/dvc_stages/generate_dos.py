import hydra
from omegaconf import DictConfig

from src.attacks.dos_generator import inject_dos_attack
from config import ATTACKS_DIR, TENSORS_VAL_PATH


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    output_tensors_path = ATTACKS_DIR / "dos_tensors.npy"
    output_labels_path = ATTACKS_DIR / "dos_labels.npy"

    inject_dos_attack(
        clean_tensors_path=TENSORS_VAL_PATH,
        output_tensors_path=output_tensors_path,
        output_labels_path=output_labels_path,
        start_row=cfg.attacks.dos.start_row,
        duration_rows=cfg.attacks.dos.duration_rows,
        target_id_col=cfg.attacks.dos.target_id_col,
    )


if __name__ == "__main__":
    main()
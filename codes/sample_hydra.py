import os
from omegaconf import DictConfig
import hydra

@hydra.main(config_path='../config/config.yaml')
def main(cfg: DictConfig) -> None:
    cwd = hydra.utils.get_original_cwd()
    print("current working directory:", cwd)
    print(cfg)
    print(cfg.lgbm.metric)


main()

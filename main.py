import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='configs', config_name='default.yaml')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(type(cfg.test.db.timeout))
    print(type(cfg.test.db.float))

if __name__ == '__main__':
    main()
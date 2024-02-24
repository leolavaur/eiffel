import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="eiffel", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()

# eiffel
Evaluation framework for FL-based intrusion detection using Flower.

## Usage

### As a tool

You can use `eiffel` as an experiment engine.
To do so, you need to provide a configuration file entitled `config.yaml`, which will be passed to Hydra.

```bash
python -m eiffel -cd path/to/workdir/
```

By default, `eiffel` will look for a Git repository, either in the current directory or in one of its parents.
The root of the repository will be used as the working directory for Hydra, meaning that the default `outputs/` and `multirun/` directories will be created there.

### As a library



## Readings for the integration with Hydra
* [Hydra\: Your Own Configuration Files — Maze documentation](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/custom_config.html "Hydra: Your Own Configuration Files — Maze documentation")
* [Hydra\: Your Own Configuration Files — Maze documentation](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/custom_config.html#hydra-custom-components "Hydra: Your Own Configuration Files — Maze documentation")
* [Experiment Configuration — Maze documentation](https://maze-rl.readthedocs.io/en/latest/workflow/experimenting.html#experimenting "Experiment Configuration — Maze documentation")
* [Configuring Experiments \| Hydra](https://hydra.cc/docs/patterns/configuring_experiments/ "Configuring Experiments | Hydra")

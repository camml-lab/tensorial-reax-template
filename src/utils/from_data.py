import functools
from typing import Optional, Union

import hydra
import jaxtyping as jt
import jraph
import omegaconf
import reax
import reax.utils


class FromData(reax.stages.EpochStage):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        strategy: reax.Strategy,
        rng: reax.Generator,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
    ):
        if dataloader is None:
            datamanager = reax.stages.common.get_datasource(datamodule, datamodule)
            dataloader = datamanager.get_loader_proxy("train_dataloader")
        else:
            datamanager = None

        super().__init__(
            "from_data",
            module=None,
            strategy=strategy,
            rng=rng,
            dataloader=dataloader,
            datamanager=datamanager,
        )
        self._cfg = cfg

    coll_dict = {label: reax.metrics.get_registry()[name] for name, label in cfg.items()}
    collection = reax.metrics.MetricCollection(coll_dict)
    results = metrics_.Evaluator(collection).evaluate(training_data)

    # Update the configuration with the values we calculated
    for name, label in from_data.items():
        value = results[label]
        from_data[name] = value.tolist() if isinstance(value, jax.Array) else value


def calculate_stats(
    from_data: omegaconf.DictConfig, training_data: reax.DataLoader[jraph.GraphsTuple]
):
    """This does an inplace update of the from_data config."""
    with_dependencies = []

    # Find those that we will come back to for a second path
    for entry in find_iterpol(from_data):
        with_dependencies.append(entry[0][0])
    with_dependencies = set(with_dependencies)

    to_calculate = {}
    for label, value in from_data.items():
        if label in with_dependencies:
            continue

        if omegaconf.OmegaConf.is_dict(value):
            stat = hydra.utils.instantiate(value, _convert_="object")
        else:
            stat = reax.metrics.get(value)

        to_calculate[label] = stat

    # Calculate the statistics
    calculated = reax.evaluate_stats(to_calculate, training_data)

    # Convert to types that can be used by omegaconf and update the configuration with the values
    calculated = {label: reax.utils.arrays.to_base(stat) for label, stat in calculated.items()}

    from_data.update(calculated)

    to_calculate = {}
    for label in with_dependencies:
        value = from_data[label]
        if omegaconf.OmegaConf.is_dict(value):
            stat = hydra.utils.instantiate(value, _convert_="object")
        else:
            stat = reax.metrics.get(value)

        to_calculate[label] = stat

    if to_calculate:
        # Calculate the dependent statistics
        calculated = reax.evaluate_stats(to_calculate, training_data)

        # Convert to types that can be used by omegaconf and update the configuration with the
        # values
        calculated = _to_omega(calculated)

        from_data.update(calculated)


@functools.singledispatch
def _to_omega(value):
    return value


@_to_omega.register
def _(value: dict) -> dict:
    return {key: _to_omega(value) for key, value in value.items()}


@_to_omega.register
def _(value: jt.Array) -> Union[int, float, list]:
    return reax.utils.arrays.to_base(value)


def find_iterpol(root, path=()):
    for key, value in root.items():
        if isinstance(value, str):
            if omegaconf.OmegaConf.is_interpolation(root, key):
                yield path, key
        elif omegaconf.OmegaConf.is_dict(value):
            yield from find_iterpol(value, (key,))

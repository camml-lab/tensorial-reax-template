"""This module exposes some commonly used metrics."""

import e3nn_jax
import reax.metrics
from tensorial import gcnn

__all__ = (
    # "EnergyPerAtomLstsq",
    "EnergyPerAtomRmse",
    "EnergyPerAtomMae",
    "ForceRmse",
)


def _convert(*args, mask=None) -> tuple:
    """Convert irreps arrays to regular arrays as not all metric operations will work on irrep
    arrays."""
    args = [entry.array if isinstance(entry, e3nn_jax.IrrepsArray) else entry for entry in args]
    if mask is not None:
        args.append(mask)
    return tuple(args)


EnergyPerAtomRmse = gcnn.graph_metric(
    reax.metrics.RootMeanSquareError.from_fun(_convert)(),
    targets="globals.energy",
    predictions="globals.predicted_energy",
    mask="globals.mask",
    normalise_by="n_node",
)

EnergyPerAtomMae = gcnn.graph_metric(
    reax.metrics.MeanAbsoluteError.from_fun(_convert)(),
    targets="globals.energy",
    predictions="globals.predicted_energy",
    mask="globals.mask",
    normalise_by="n_node",
)

ForceRmse = gcnn.graph_metric(
    reax.metrics.RootMeanSquareError.from_fun(_convert)(),
    targets="nodes.forces",
    predictions="nodes.predicted_forces",
    mask="nodes.mask",
)


# EnergyPerAtomLstsq = metrics.from_fun(
#     lambda graph: metrics.RootMeanSquareError.create(
#         graph.globals[keys.SPECIES],
#         graph.globals[atomic.TOTAL_ENERGY],
#     )
# )

"""Template to get started using REAX with tensorial."""

import reax

from . import metrics, utils

reax.metrics.get_registry().register_many(
    {
        "atomic/energy_per_atom_rmse": metrics.EnergyPerAtomRmse,
        "atomic/energy_per_atom_mae": metrics.EnergyPerAtomMae,
        "atomic/force_rmse": metrics.ForceRmse,
    }
)

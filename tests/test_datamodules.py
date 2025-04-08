from pathlib import Path

import jraph
import numpy as np
import pytest
import reax

from trt.data import qm9_datamodule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_datamodule(batch_size: int) -> None:
    """Tests `MnistDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = qm9_datamodule.Qm9DataModule(r_max=3.0, data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, qm9_datamodule.Qm9DataModule.FILENAME).exists()

    stage = reax.stages.Train(
        None, reax.data.create_manager(datamodule=dm), None, [], reax.Generator()
    )
    dm.setup(stage)
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 133_885

    batch: jraph.GraphsTuple = next(iter(dm.train_dataloader()))[0]
    assert len(batch.n_node) == batch_size + 1
    assert batch.nodes["positions"].dtype == np.float64
    assert batch.nodes["atomic_numbers"].dtype == np.int64

import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

from nn import PaiNN

import torch
import torchmetrics
import pytorch_lightning as pl

if __name__ == '__main__':

    # prepare working directory
    qm9tut = './qm9tut'
    if not os.path.exists('qm9tut'):
        os.makedirs(qm9tut)

    # QM9 data
    os.system('rm ' + os.path.join(qm9tut, "split.npz"))
    qm9data = QM9(
        './qm9.db',
        batch_size=100,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=False, # set to false, when not using a GPU
        load_properties=[QM9.U0], #only load U0 property, i.e. inner energy at 0K
    )
    qm9data.prepare_data()
    qm9data.setup()

    # some insides
    atomrefs = qm9data.train_dataset.atomrefs
    print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')
    print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
    print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')

    means, stddevs = qm9data.get_stats(
        QM9.U0, divide_by_atoms=True, remove_atomref=True
    )
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())


    # Model Setup (QM9)
    cutoff = 5.  # Angstrom
    n_atom_basis = 128
    n_interactions = 3

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    painn = PaiNN(
        n_atom_basis=       n_atom_basis,
        n_interactions=     n_interactions,
        radial_basis=       spk.nn.radial.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=          spk.nn.cutoff.CosineCutoff(cutoff=cutoff)
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
    )

    # Model Output
    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    # Training Task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    # Training + Monitoring + Logging
    logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm9tut,
        max_epochs=3, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=qm9data)
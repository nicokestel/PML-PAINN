import os
import schnetpack as spk
from schnetpack.datasets import MD17
import schnetpack.transform as trn

from nn import PaiNN
from data_loader import load_data

import torch
import torchmetrics
import pytorch_lightning as pl

def run():
    work_dir = './md17_ef'

    md17data = load_data('md17',
                         molecule='ethanol',
                         transformations=[
                             trn.ASENeighborList(cutoff=5.),
                             trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
                             trn.MatScipyNeighborList(cutoff=5.),
                             trn.CastTo32()
                         ],
                         n_train=1000,  # 100000
                         n_val=1000,
                         batch_size=10,
                         work_dir=work_dir)
    
    # Model Setup (MD17)
    cutoff = 5.  # Angstrom
    n_atom_basis = 3  # 128
    n_interactions = 3  # 64

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    painn = PaiNN(
        n_atom_basis=       n_atom_basis,
        n_interactions=     n_interactions,
        radial_basis=       spk.nn.radial.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=          spk.nn.cutoff.CosineCutoff(cutoff=cutoff)
    )
    pred_e = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy, aggregation_mode='sum')
    pred_f = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[pairwise_distance],
        output_modules=[pred_e, pred_f],
        postprocessors=[trn.CastTo64(), trn.AddOffsets(property='energy', add_mean=True)]
    )

    # Model Output
    output_e = spk.task.ModelOutput(
        name=MD17.energy,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError(),
            "RMSE": torchmetrics.MeanSquaredError(squared=False)
        }
    )

    output_f = spk.task.ModelOutput(
        name=MD17.forces,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError(),
            "RMSE": torchmetrics.MeanSquaredError(squared=False)
        }
    )

    # Training Task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_e, output_f],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3},  # "weight_decay": 0.01 by default
        scheduler_cls=spk.train.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 50, "smoothing_factor": 0.9},
        scheduler_monitor="val_loss"
    )

    # Training + Monitoring + Logging
    logger = pl.loggers.TensorBoardLogger(save_dir=work_dir)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(work_dir, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=150
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=work_dir,
        max_epochs=3, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=md17data)
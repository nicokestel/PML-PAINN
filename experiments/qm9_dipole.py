import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

from nn import PaiNN
from data_loader import load_data

import torch
import torchmetrics
import pytorch_lightning as pl

#if __name__ == '__main__':
def run():
    work_dir = '/home/space/datasets/qm9_dipole'

    qm9dipole = load_data('qm9',
                          transformations=[
                              trn.SubtractCenterOfMass(),
                              trn.MatScipyNeighborList(cutoff=5.),
                              trn.CastTo32()
                          ],
                          n_train=1000,  # 100000
                          n_val=1000,
                          batch_size=100,
                          work_dir=work_dir)

    # some insides
    #atomrefs = qm9dipole.train_dataset.atomrefs
    #print(atomrefs)
    #print('dipole_moment of hyrogen:', atomrefs[QM9.mu][1].item(), 'D')
    #print('dipole_moment of carbon:', atomrefs[QM9.mu][6].item(), 'D')
    #print('dipole_moment of oxygen:', atomrefs[QM9.mu][8].item(), 'D')

    #means, stddevs = qm9dipole.get_stats(
    #    QM9.mu, divide_by_atoms=True, remove_atomref=True
    #)
    #print('Mean dipole momentum / atom:', means.item())
    #print('Std. dev. dipole momentum / atom:', stddevs.item())


    # Model Setup (QM9)
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
    pred_mu = spk.atomistic.DipoleMoment(n_in=n_atom_basis, predict_magnitude=True, use_vector_representation=True)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[pairwise_distance],
        output_modules=[pred_mu],
        postprocessors=[trn.CastTo64()]
    )

    # Model Output
    output_mu = spk.task.ModelOutput(
        name=QM9.mu,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError(),
            "RMSE": torchmetrics.MeanSquaredError(squared=False)
        }
    )
    print(output_mu.target_property)

    # Training Task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_mu],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 5e-4},  # "weight_decay": 0.01 by default
        scheduler_cls=spk.train.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 5, "smoothing_factor": 0.9},
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
            patience=30
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=work_dir,
        max_epochs=3, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=qm9dipole)
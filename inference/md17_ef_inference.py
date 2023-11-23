import os

import torch
from torch.utils.data import random_split

import numpy as np
from ase import Atoms

import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets.md17 import MD17

from data_loader import load_data


def run(path_to_model, path_to_data_dir, molecule='ethanol'):
    # set device
    device = torch.device("cuda")

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)

    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    # load dataset
    dataset = load_data('md17',
                        molecule=molecule,
                        transformations=[
                            trn.SubtractCenterOfMass(),
                            trn.RemoveOffsets(MD17.energy, remove_mean=True),
                            trn.MatScipyNeighborList(cutoff=5.),
                            trn.CastTo32()
                        ],
                        n_train=950,  # 950
                        n_val=50,  # 50
                        batch_size=10,
                        work_dir=path_to_data_dir)

    # create atoms object from dataset
    splits = random_split(dataset.test_dataset, [1/3, 1/3, 1/3])

    # compute MAE over 3 random splits
    energy_avg_mae, forces_avg_mae = 0.0
    for split in splits:
        atoms = Atoms(
            numbers=split[spk.properties.Z], positions=split[spk.properties.R]
        )

        # convert atoms to SchNetPack inputs and perform prediction
        inputs = converter(atoms)
        results = best_model(inputs)

        energy_avg_mae += torch.nn.L1Loss(results['energy'], inputs['energy'])
        forces_avg_mae += torch.nn.L1Loss(results['forces'], inputs['forces'])
    
    energy_avg_mae /= len(splits)
    forces_avg_mae /= len(splits)

    print('AVG (over 3 random splits) MAE on energy: {:.3f}'.format(energy_avg_mae))
    print('AVG (over 3 random splits) MAE on forces: {:.3f}'.format(forces_avg_mae))

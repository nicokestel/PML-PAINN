import os
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import numpy as np
from ase import Atoms

import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets.md17 import MD17
# from schnetpack.data.loader import AtomsLoader
from schnetpack.interfaces.ase_interface import SpkCalculator

from schnetpack.units import convert_units

from data_loader import load_data


def run(path_to_model, path_to_data_dir, molecule='ethanol'):
    # set device
    device = torch.device("cpu")

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)
    best_model.eval()

    # set up converter
    # converter = spk.interfaces.AtomsConverter(
    #     neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    # )

    calculator = spk.interfaces.SpkCalculator(
        path_to_model, trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
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
    
    #print(dataset.test_idx)

    # create atoms object from dataset
    # splits = random_split(dataset.test_dataset, [1/3, 1/3, 1/3])
    n_test = i = 10
    # splits = dataset.test_dataset[:n_test//3], dataset.test_dataset[n_test//3 : 2*n_test//3], dataset.test_dataset[2*n_test//3:]

    # compute MAE
    energy_avg_mae, forces_avg_mae = 0.0, 0.0
    for batch in dataset.test_dataloader():
        if i == 0:
            break

        print(batch['energy'])
        results = best_model(copy.deepcopy(batch))

        # convert units
        # forces stays same
        results['energy'] *= convert_units("kcal/mol", "eV")

        energy_avg_mae += F.l1_loss(results['energy'], batch['energy'])
        forces_avg_mae += F.l1_loss(results['forces'], batch['forces'])
        i -= 1


    energy_avg_mae /= n_test
    forces_avg_mae /= n_test

    print('MAE on energy: {:.3f}'.format(
        energy_avg_mae))
    print('MAE on forces: {:.3f}'.format(
        forces_avg_mae))
